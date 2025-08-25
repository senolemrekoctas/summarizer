
import fs from 'fs/promises';
import path from 'path';


const TR_STOPWORDS = new Set([
  'acaba','ama','aslında','az','bazı','belki','biri','birkaç','birçok','böyle','bu','çok','çünkü','da','daha','de','defa','diye','eğer','en','gibi','hem','hep','hepsi','her','hiç','ile','ise','için','kadar','ki','kim','mı','mi','mu','mü','nasıl','ne','neden','nerde','nerede','nereye','niçin','niye','o','sanki','siz','şey','sonra','şu','tüm','ve','veya','ya','yani','olarak','üzere','fakat','ancak','herhangi','hiçbir','herkes','her şey','hemen','artık','yine','bile','bazen','özellikle','olsa','olduğu','olduğunu','olan','olanlar','olabilir','olmak','etmek','yapmak','var','yok','göre','kendi','kendisi','arada','aynı','bana','bende','beni','benim','biz','bizim','sizin','sizi','sizde','sana','seni','senin','onun','onlar','onları','onların','şimdi','bugün','yarın','dün','bir','iki','üç','dört','beş','altı','yedi','sekiz','dokuz','on'
]);


function parseArgs(argv) {
  const out = {};
  for (let i = 2; i < argv.length; i++) {
    const a = argv[i];
    if (a.startsWith('--')) {
      const key = a.slice(2);
      const v = argv[i + 1];
      if (!v || v.startsWith('--')) {
        out[key] = true;
      } else {
        out[key] = v;
        i++;
      }
    }
  }
  return out;
}

async function readInput(args) {
  if (args.text) return args.text.toString();
  if (args.file) {
    const p = path.resolve(process.cwd(), args.file);
    return fs.readFile(p, 'utf-8');
  }
  throw new Error('Metin girişi bulunamadı. --text veya --file kullanın.');
}


function splitSentences(text) {
  return text
    .replace(/\s+/g, ' ')
    .split(/(?<=[\.\!\?])\s+|\n+/)
    .map(s => s.trim())
    .filter(Boolean);
}

function tokenize(text) {
  const cleaned = text
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s]/gu, ' ')
    .replace(/\s+/g, ' ')
    .trim();
  return cleaned.split(' ').filter(w => w && w.length > 1 && !TR_STOPWORDS.has(w));
}


function buildTfidfVectors(sentences) {
  const docs = sentences.map(tokenize);
  const vocab = new Map();
  for (const doc of docs) for (const w of doc) if (!vocab.has(w)) vocab.set(w, vocab.size);
  const V = vocab.size, N = docs.length;

  const tf = docs.map(doc => {
    const vec = new Float32Array(V);
    const counts = new Map();
    for (const w of doc) counts.set(w, (counts.get(w) || 0) + 1);
    const len = doc.length || 1;
    for (const [w, c] of counts.entries()) vec[vocab.get(w)] = c / len;
    return vec;
  });

  const df = new Uint32Array(V);
  for (const doc of docs) {
    const seen = new Set(doc);
    for (const w of seen) df[vocab.get(w)]++;
  }

  const idf = new Float32Array(V);
  for (let j = 0; j < V; j++) idf[j] = Math.log((1 + N) / (1 + df[j])) + 1;

  return tf.map(vec => {
    const out = new Float32Array(V);
    for (let j = 0; j < V; j++) out[j] = vec[j] * idf[j];
    return out;
  });
}

function cosine(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    const x = a[i], y = b[i];
    dot += x * y; na += x * x; nb += y * y;
  }
  if (!na || !nb) return 0;
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

function buildSimilarityMatrix(vectors) {
  const n = vectors.length;
  const M = Array.from({ length: n }, () => new Float32Array(n));
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const sim = cosine(vectors[i], vectors[j]);
      M[i][j] = sim; M[j][i] = sim;
    }
  }
  for (let i = 0; i < n; i++) M[i][i] = 0;
  return M;
}

function pageRank(M, { d = 0.85, tol = 1e-6, maxIter = 100 } = {}) {
  const n = M.length;
  if (n === 0) return [];
  const S = Array.from({ length: n }, () => new Float32Array(n));
  for (let i = 0; i < n; i++) {
    let sum = 0; for (let j = 0; j < n; j++) sum += M[i][j];
    if (sum === 0) for (let j = 0; j < n; j++) S[i][j] = 1 / n;
    else for (let j = 0; j < n; j++) S[i][j] = M[i][j] / sum;
  }
  let r = new Float32Array(n).fill(1 / n);
  const teleport = (1 - d) / n;
  for (let it = 0; it < maxIter; it++) {
    const rNew = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      let acc = 0; for (let j = 0; j < n; j++) acc += r[j] * S[j][i];
      rNew[i] = teleport + d * acc;
    }
    let diff = 0; for (let i = 0; i < n; i++) diff += Math.abs(rNew[i] - r[i]);
    r = rNew; if (diff < tol) break;
  }
  return Array.from(r);
}

function extractiveSummarize(text, { ratio = 0.2, max = null } = {}) {
  const sents = splitSentences(text);
  if (sents.length <= 2) return text.trim();
  const vectors = buildTfidfVectors(sents);
  const M = buildSimilarityMatrix(vectors);
  const scores = pageRank(M);
  const idx = scores.map((s, i) => [s, i]).sort((a, b) => b[0] - a[0]);
  const k = Math.max(1, Math.min(max ?? Math.ceil(sents.length * ratio), sents.length));
  const chosen = idx.slice(0, k).map(([, i]) => i).sort((a, b) => a - b);
  return chosen.map(i => sents[i]).join(' ');
}

// -----------------------------
// Ollama HTTP (streamsiz)
// -----------------------------
async function ollamaChat(messages, { model = 'llama3.2:3b', temperature = 0.1, top_p = 0.9, repeat_penalty = 1.2 } = {}) {
  const res = await fetch('http://localhost:11434/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model, stream: false, options: { temperature, top_p, repeat_penalty }, messages })
  });
  if (!res.ok) {
    const t = await res.text().catch(() => '');
    throw new Error('Ollama HTTP error ' + res.status + ': ' + t);
  }
  const data = await res.json();
  return data?.message?.content?.trim() ?? '';
}

function chunkText(text, maxChars = 8000, overlap = 200) {
  const chunks = [];
  let i = 0;
  while (i < text.length) {
    const end = Math.min(i + maxChars, text.length);
    const chunk = text.slice(i, end);
    chunks.push(chunk);
    if (end === text.length) break;
    i = Math.max(0, end - overlap);
  }
  return chunks;
}

async function abstractiveSummarize(text, { model = 'llama3.2:3b', sentences = 5, maxWords = 80 } = {}) {

  const chunks = chunkText(text);
  const partials = [];
  for (const ch of chunks) {
    const sys = { role: 'system', content:
      'Sen Türkçe konuşan profesyonel bir özetleyicisin. ' +
      'Sadece verilen metne dayalı, kısa ve doğru özet yaz. Yeni bilgi ekleme.' };
    const usr = { role: 'user', content:
      `Bu parçayı ${sentences} madde/satırda, her madde en fazla ~${Math.ceil(maxWords / sentences)} kelime olacak şekilde özetle. ` +
      'Sadece maddeleri ver, giriş/cıkış cümlesi yazma.\n\n' + ch };
    const out = await ollamaChat([sys, usr], { model, temperature: 0.1, repeat_penalty: 1.2 });
    partials.push(out);
  }


  const combined = partials.join('\n');
  const sys2 = { role: 'system', content:
    'Türkçe metin özetleyicisin. Yeni bilgi ekleme, sadece verilen içerikten yararlan.' };
  const usr2 = { role: 'user', content:
    `Aşağıdaki maddeleri tek, akıcı bir paragrafta topla. TAM ${sentences} cümle yaz ve toplam ${maxWords} kelimeyi geçme. ` +
    'Başlık veya madde işareti kullanma.\n\n' + combined };

  let final = await ollamaChat([sys2, usr2], { model, temperature: 0.1 });

  const sys3 = { role: 'system', content:
    'Türkçe metin düzelticisin. Yazım ve dil bilgisi hatalarını düzelt, daha akıcı yap. ' +
    'Anlamı değiştirme, yeni bilgi ekleme. Paragraf biçimini koru.' };
  final = await ollamaChat([sys3, { role: 'user', content: final }], { model, temperature: 0.0 });

  return final.trim();
}


async function main() {
  const args = parseArgs(process.argv);
  const mode = (args.mode || '').toLowerCase();

  if (!['extractive','abstractive'].includes(mode)) {
    console.log('Kullanım:');
    console.log('  node summarizer_cli.mjs --mode extractive --ratio 0.25 --file input.txt');
    console.log('  node summarizer_cli.mjs --mode extractive --text "metin" --max 5');
    console.log('  node summarizer_cli.mjs --mode abstractive --file input.txt --sentences 5 --model llama3.2:3b [--maxWords 80]');
    process.exit(1);
  }

  const text = await readInput(args);

  if (mode === 'extractive') {
    const ratio = args.ratio ? parseFloat(args.ratio) : 0.2;
    const max = args.max ? parseInt(args.max, 10) : null;
    const summary = extractiveSummarize(text, { ratio, max });
    console.log('\n--- ÖZET (Extractive) ---\n');
    console.log(summary);
    console.log('\n-------------------------\n');
  } else {
    const model = args.model || 'llama3.2:3b'; // istersen 'phi3:mini'
    const sentences = args.sentences ? parseInt(args.sentences, 10) : 3;
    const maxWords = args.maxWords ? parseInt(args.maxWords, 10) : 60;
    const summary = await abstractiveSummarize(text, { model, sentences, maxWords });
    console.log('\n--- ÖZET (Abstractive / Ollama) ---\n');
    console.log(summary);
    console.log('\n-----------------------------------\n');
  }
}

main().catch(err => {
  console.error('[Hata]', err.message);
  process.exit(1);
});
 //node .\summarizer_cli.mjs --mode extractive --file .\input.txt --ratio 0.25
