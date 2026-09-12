// Drive chat.html's real logic against a live coordinator, without a browser.
//
// Extracts the page's module script, stubs only the DOM surface it touches, and
// streams a real turn through it -- so the SSE parsing, the topology draw and
// the stats table are exercised as written. node --check proves a page parses;
// this proves it works.
//
// Needs a coordinator with its layers covered:
//   node src/server.js &  node sim_bird.mjs solo &  node test/chat_ui.test.mjs
import {readFileSync} from 'fs';
import path from 'path';
import {fileURLToPath} from 'url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const BASE = process.env.FLOCK_URL || 'http://127.0.0.1:8000';
const html = readFileSync(path.join(ROOT, 'web/chat.html'), 'utf8');
const src = /<script type="module">([\s\S]*?)<\/script>/.exec(html)[1];

// --- minimal DOM -----------------------------------------------------------
const nodes = new Map();
let created = 0;
function mk(tag) {
  created++;
  const n = {
    tagName: String(tag).toUpperCase(), className: '', id: '', title: '',
    children: [], style: {}, _text: '', _html: '',
    classList: {
      _s: new Set(),
      add(...c) { c.forEach(x => this._s.add(x)); },
      remove(...c) { c.forEach(x => this._s.delete(x)); },
      contains(c) { return this._s.has(c); },
    },
    get textContent() { return this._text; },
    set textContent(v) { this._text = String(v); this.children.length = 0; },
    get innerHTML() { return this._html; },
    set innerHTML(v) { this._html = String(v); },
    append(...kids) {
      for (const k of kids) {
        this.children.push(k);
        if (k && k.id) nodes.set(k.id, k);
      }
    },
    remove() {},
    querySelector(sel) {
      const want = sel.replace('.', '');
      const walk = n => {
        for (const c of n.children || []) {
          if (c.classList?.contains(want)) return c;
          const d = walk(c);
          if (d) return d;
        }
        return null;
      };
      return walk(this);
    },
    addEventListener() {}, focus() {},
    scrollHeight: 0, scrollTop: 0, value: '', disabled: false, rows: 1,
  };
  return n;
}
for (const id of ['sub', 'pipe', 'convo', 'empty', 'ctx', 'diag', 'q',
                  'send', 'clear', 'f']) {
  const n = mk('div'); n.id = id; nodes.set(id, n);
}
globalThis.document = {
  getElementById: id => nodes.get(id) || null,
  createElement: mk,
  addEventListener() {},
};
globalThis.matchMedia = () => ({matches: false});
globalThis.setInterval = () => 0;   // no background polling in the harness

// --- run the page ----------------------------------------------------------
// The page fetches same-origin paths; point them at the coordinator under test.
const rewritten = src.replace(/fetch\('\//g, `fetch('${BASE}/`)
  // The page reports any poll() failure as "unreachable"; show the real reason.
  .replace('} catch {\n    $(\'sub\').innerHTML',
           '} catch (e) {\n    console.log(\'  poll threw:\', e.message);\n    $(\'sub\').innerHTML');
const mod = `data:text/javascript;base64,${Buffer.from(rewritten).toString('base64')}`;
await import(mod);
await new Promise(r => setTimeout(r, 800));   // let poll() land

const flat = n => {
  const out = [];
  const walk = x => { if (x._text) out.push(x._text); (x.children || []).forEach(walk); };
  walk(n);
  return out;
};
const sub = nodes.get('sub');
console.log('header   :', (sub.innerHTML || sub.textContent).replace(/<[^>]+>/g, ''));
console.log('context  :', nodes.get('ctx').textContent);
const pipe = nodes.get('pipe');
console.log('topology :', pipe.children.length, 'cells');
console.log('cells    :', flat(pipe).join(' | '));
let fails = 0;
const check = (name, cond) => {
  console.log(`  ${cond ? 'ok  ' : 'FAIL'} ${name}`);
  if (!cond) fails++;
};
// The expected cell count comes from the LIVE flock rather than a constant. The
// topology used to be fixed by web/flock.json, so `=== 9` (coordinator + 4 birds
// + 4 arrows) was a safe hardcode; now the split is chosen at startup from the
// GGUF header, so a 1-bird flock renders 3 cells and a 4-bird one renders 9.
// Deriving it means this asserts the page renders WHAT IS THERE -- which is the
// actual claim -- instead of failing whenever the coordinator is configured
// differently.
const live = await (await fetch(`${BASE}/status`)).json();
const nBirds = live.birds.length;
check(`coordinator + ${nBirds} bird(s) + ${nBirds} arrow(s) rendered`,
  pipe.children.length === 1 + 2 * nBirds);
check('every device shows its layer range',
  flat(pipe).filter(t => /^layers /.test(t)).length === 1 + nBirds);
check('header reports the flock ready', /layers/.test(sub.innerHTML));

// --- drive one real turn through the page's own send() ----------------------
console.log('\nstreaming a turn through the page logic…');
const convo = nodes.get('convo');
const before = convo.children.length;
// send() is module-scoped, so reach it the way the form does.
nodes.get('q').value = 'Say hello in three words.';
await nodes.get('f').onsubmit({preventDefault() {}});
for (let i = 0; i < 600 && nodes.get('send').disabled; i++)
  await new Promise(r => setTimeout(r, 200));
await new Promise(r => setTimeout(r, 400));

const msgs = convo.children.slice(before);
console.log(`messages added: ${msgs.length}`);
for (const m of msgs) console.log(`  [${m.className}] ${flat(m).join(' ').slice(0, 150)}`);
check('a user message and a reply were added', msgs.length === 2);
check('the reply is not an error', !msgs.some(m => /err/.test(m.className)));
const reply = flat(msgs[1] || {children: []}).join(' ');
check('the reply carries generated text', reply.length > 20);
check('per-turn stats summarise tok/s', /tok\/s/.test(reply));
check('the stats table has a prefill row', /prefill/.test(reply));
console.log(`\ndom nodes created: ${created}`);
console.log(fails ? `${fails} CHECKS FAILED` : 'ALL CHECKS PASSED');
process.exit(fails ? 1 : 0);
