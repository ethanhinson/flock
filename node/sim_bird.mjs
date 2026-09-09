// Stand-in bird against the Node coordinator (websocket path).
import ort from 'onnxruntime-node';
import WebSocket from 'ws';
import {pack, unpack} from '../web/js/wire.mjs';
process.chdir('/Users/ethanhinson/dev/flock');
const label = process.argv[2] || 'sim';
const j = await (await fetch('http://127.0.0.1:8000/join', {
  method:'POST', headers:{'Content-Type':'application/json'},
  body: JSON.stringify({label})})).json();
if (j.error) { console.log(j.error); process.exit(1); }
console.log(`bird ${j.slot}: layers ${j.start}-${j.end}`);
const s = await ort.InferenceSession.create(`web/shard${j.slot}.onnx`);
let past = null;
const empty = () => new ort.Tensor('float32', new Float32Array(0), [1,j.kv_heads,0,j.head_dim]);
const ws = new WebSocket('ws://127.0.0.1:8000/ws');
ws.on('open', () => { ws.send(JSON.stringify({peer_id:j.peer_id,label})); console.log('ws open'); });
ws.on('message', async (data, isBinary) => {
  if (!isBinary) return;
  const ab = data.buffer.slice(data.byteOffset, data.byteOffset+data.byteLength);
  const {data:h, meta:m} = unpack(ab);
  if (m.reset) past = null;
  const feed = {
    hidden: new ort.Tensor('float32', h, [1, m.seq, j.hidden]),
    position_ids: new ort.Tensor('int64',
      BigInt64Array.from({length:m.seq},(_,i)=>BigInt(m.offset+i)), [1,m.seq]),
  };
  for (let i=0;i<j.n_layers;i++){
    feed[`past_k${i}`] = past?past[`new_k${i}`]:empty();
    feed[`past_v${i}`] = past?past[`new_v${i}`]:empty();
  }
  const t0=performance.now();
  const out = await s.run(feed);
  const ms=+(performance.now()-t0).toFixed(1);
  past={}; for(let i=0;i<j.n_layers;i++){past[`new_k${i}`]=out[`new_k${i}`];past[`new_v${i}`]=out[`new_v${i}`];}
  ws.send(Buffer.from(pack(out.output.data,{seq:m.seq,hidden:j.hidden,offset:m.offset})));
  ws.send(JSON.stringify({t:'stats',ms,transport:'ws'}));
});
