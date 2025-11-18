// Express app wiring
import express from 'express';
import helmet from 'helmet';
import cors from 'cors';
import morgan from 'morgan';
import { createServer } from "node:http";
import dotenv from 'dotenv';
import multer from 'multer';
import axios from 'axios';
import FormData from 'form-data';
import sharp from 'sharp';
import ExifParser from 'exif-parser';
dotenv.config();
const PORT = process.env.PORT || 5000;
const app = express();
const server = createServer(app);


// Security, logging, parsing
app.use(helmet());
// Allow all CORS origins (dev-friendly)
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use(morgan(process.env.NODE_ENV === 'production' ? 'combined' : 'dev'));

// Add minimal dbConnect stub to allow server start
const dbConnect = async () => {};

// File upload handler
const upload = multer({ storage: multer.memoryStorage() });

// --- helpers (local forensic)
const toGray = (data) => {
  const gray = new Float32Array(data.length / 3);
  for (let i = 0, gi = 0; i < data.length; i += 3, gi += 1) {
    gray[gi] = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
  }
  return gray;
};
const mean = (arr) => {
  let s = 0;
  for (let i = 0; i < arr.length; i++) s += arr[i];
  return s / arr.length;
};
const std = (arr) => {
  const m = mean(arr);
  let v = 0;
  for (let i = 0; i < arr.length; i++) {
    const d = arr[i] - m;
    v += d * d;
  }
  return Math.sqrt(v / arr.length);
};
const corr = (a, b) => {
  const ma = mean(a), mb = mean(b);
  let num = 0, da = 0, db = 0;
  for (let i = 0; i < a.length; i++) {
    const xa = a[i] - ma;
    const xb = b[i] - mb;
    num += xa * xb;
    da += xa * xa;
    db += xb * xb;
  }
  const denom = Math.sqrt(da * db);
  if (denom < 1e-8) return 0;
  return num / denom;
};
const tanhNorm = (x, scale) => Math.tanh(x * scale);
const blockStd = (mat, width, height, block = 8) => {
  const variances = [];
  for (let y = 0; y <= height - block; y += block) {
    for (let x = 0; x <= width - block; x += block) {
      let s = 0, s2 = 0, n = block * block;
      for (let yy = 0; yy < block; yy++) {
        for (let xx = 0; xx < block; xx++) {
          const v = mat[(y + yy) * width + (x + xx)];
          s += v;
          s2 += v * v;
        }
      }
      const m = s / n;
      variances.push(Math.max(0, s2 / n - m * m));
    }
  }
  const m = mean(variances);
  let v = 0;
  for (let i = 0; i < variances.length; i++) {
    const d = variances[i] - m;
    v += d * d;
  }
  return Math.sqrt(v / Math.max(variances.length, 1));
};

async function forensicAnalyzeBuffer(buffer) {
  // normalize to manageable size
  const base = sharp(buffer)
    .resize(256, 256, { fit: 'inside' })
    .toColourspace('srgb') // ensure 3-channel sRGB
    .removeAlpha();
  const { data, info } = await base.raw().toBuffer({ resolveWithObject: true });
  const width = info.width;
  const height = info.height;

  const gray = toGray(data);

  // blur and residual (noise proxy)
  const blurBuf = await base.clone().blur(1.5).raw().toBuffer();
  const blurGray = toGray(blurBuf);
  const residual = new Float32Array(gray.length);
  for (let i = 0; i < gray.length; i++) residual[i] = gray[i] - blurGray[i];

  // laplacian (high-frequency proxy)
  const laplacianKernel = {
    width: 3,
    height: 3,
    kernel: [-1, -1, -1, -1, 8, -1, -1, -1, -1],
  };
  const lapBuf = await base.clone().convolve(laplacianKernel).raw().toBuffer();
  const lapGray = toGray(lapBuf);

  const residStd = std(residual);
  const lapStd = std(lapGray);

  const noiseScore = (() => {
    const residNorm = tanhNorm(residStd / 255, 10);
    const lapNorm = tanhNorm(lapStd / 255, 5);
    return Number(Math.min(1, Math.max(0, 0.55 * residNorm + 0.45 * lapNorm)));
  })();

  const frequencyScore = (() => {
    const hfEnergy = lapStd / 255;
    const score = tanhNorm(hfEnergy, 2.5);
    return Number(Math.min(1, Math.max(0, score)));
  })();

  // recompress and measure error spread (compression proxy)
  const reencoded = await sharp(buffer).jpeg({ quality: 85 }).toBuffer();
  const { data: compData } = await sharp(reencoded)
    .resize(width, height)
    .toColourspace('srgb') // keep sRGB
    .removeAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
  const compGray = toGray(compData);
  const error = new Float32Array(gray.length);
  for (let i = 0; i < gray.length; i++) error[i] = Math.abs(gray[i] - compGray[i]);
  const spread = blockStd(error, width, height, 8);
  const compressionScore = Number(Math.min(1, Math.max(0, tanhNorm(spread / 255, 8))));

  // color correlation
  const r = new Float32Array(gray.length);
  const g = new Float32Array(gray.length);
  const b = new Float32Array(gray.length);
  for (let i = 0, gi = 0; i < data.length; i += 3, gi += 1) {
    r[gi] = data[i];
    g[gi] = data[i + 1];
    b[gi] = data[i + 2];
  }
  const rg = corr(r, g);
  const rb = corr(r, b);
  const gb = corr(g, b);
  const colorCorrScore = Number(Math.min(1, Math.max(0, (rg + rb + gb) / 3 * 0.5 + 0.5)));

  // combine weights similar to Python
  const wFreq = 0.35, wNoise = 0.3, wComp = 0.2, wColor = 0.15;
  const forensicScore = Number(Math.min(1, Math.max(0, wFreq * frequencyScore + wNoise * noiseScore + wComp * compressionScore + wColor * colorCorrScore)));

  return {
    forensic_score: forensicScore,
    frequency: Number(frequencyScore),
    noise: Number(noiseScore),
    compression: Number(compressionScore),
    color_corr: Number(colorCorrScore),
    width,
    height,
  };
}

function hasExif(buffer) {
  try {
    const parser = ExifParser.create(buffer);
    const result = parser.parse();
    return !!result.tags && Object.keys(result.tags).length > 0;
  } catch {
    return false;
  }
}

// JSON endpoint returning hybrid response for charts
app.post('/api/analyze', upload.any(), async (req, res) => {
  try {
    const file = Array.isArray(req.files) && req.files.length ? req.files[0] : null;
    if (!file) {
      return res.status(400).json({
        error: 'No file provided. Send multipart/form-data with a file field.',
      });
    }

    // Map env from either SIGHTENGINE_* or API_*
    const apiUser = process.env.SIGHTENGINE_USER ?? process.env.API_USER;
    const apiSecret = process.env.SIGHTENGINE_SECRET ?? process.env.API_SECRET;
    const hasSightengineCreds = !!(apiUser && apiSecret);

    // Local forensic
    const forensic = await forensicAnalyzeBuffer(file.buffer);
    const exifPresent = hasExif(file.buffer);
    const metadataAdjustment = exifPresent ? 0 : 10;
    const forensicAiPct = (1.0 - forensic.forensic_score) * 100.0;
// 
    // Compute local component once
    const localComponent = forensicAiPct + metadataAdjustment;
    let finalAiScore = Math.max(0, Math.min(100, localComponent));

    // Try external model if credentials present
    let apiResult = {};
    let apiAiScore = 0.0;

    if (hasSightengineCreds) {
      const form = new FormData();
      form.append('media', file.buffer, {
        filename: file.originalname || 'upload.jpg',
        contentType: file.mimetype || 'application/octet-stream'
      });
      form.append('models', 'genai');
      form.append('api_user', apiUser);
      form.append('api_secret', apiSecret);

      try {
        const resp = await axios.post('https://api.sightengine.com/1.0/check.json', form, {
          headers: form.getHeaders(),
          timeout: 30000
        });

        apiResult = resp.data || {};
        apiAiScore = apiResult?.type?.ai_generated ? apiResult.type.ai_generated * 100.0 : 0.0;

        const apiW = 0.7, localW = 0.3;
        finalAiScore = Math.max(0, Math.min(100, apiAiScore * apiW + localComponent * localW));
      } catch (extErr) {
        apiResult = { error: 'external_call_failed', details: extErr?.response?.data ?? { message: extErr.message } };
      }
    }

    // Breakdown and response
    const metrics = ['Sightengine', 'Frequency', 'Noise', 'Compression', 'Color', 'Metadata'];
    const scores = [
      apiAiScore,
      (1 - forensic.frequency) * 100,
      (1 - forensic.noise) * 100,
      (1 - forensic.compression) * 100,
      (1 - forensic.color_corr) * 100,
      metadataAdjustment
    ];

    return res.status(200).json({
      status: 'success',
      sightengine: apiResult,
      local: forensic,
      metadata: { exif_present: exifPresent },
      hybrid: {
        api_ai_score: Number(apiAiScore),
        forensic_ai_pct: Number(forensicAiPct),
        metadata_adjustment: Number(metadataAdjustment),
        final_ai_score: Number(finalAiScore)
      },
      breakdown: { metrics, scores }
    });
  } catch (error) {
    const details = error.response?.data || { message: error.message, stack: error.stack };
    return res.status(502).json({ error: 'Analysis failed', details });
  }
});

const startServer = async () => {
  try {
    await dbConnect();
    server.listen(PORT, () => {
      console.log("Server running on http://0.0.0.0:5233");
      console.log(`✅ Server running on port ${PORT}`);
    });
  } catch (error) {
    console.error("❌ Database connection failed", error);
    process.exit(1);
  }
};
startServer();

