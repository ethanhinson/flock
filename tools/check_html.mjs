// node --check cannot read HTML, so pull each <script type="module"> out of the
// page and check it as a module. Run: node check_html.mjs ../web/*.html
import {readFileSync, writeFileSync, mkdtempSync} from 'fs';
import {execFileSync} from 'child_process';
import {tmpdir} from 'os';
import path from 'path';

let bad = 0;
for (const f of process.argv.slice(2)) {
  const html = readFileSync(f, 'utf8');
  const re = /<script\b([^>]*)>([\s\S]*?)<\/script>/gi;
  let m, i = 0;
  while ((m = re.exec(html))) {
    if (/\bsrc=/.test(m[1])) continue;         // external, nothing inline to check
    const dir = mkdtempSync(path.join(tmpdir(), 'flockchk-'));
    const out = path.join(dir, `s${i}.mjs`);
    writeFileSync(out, m[2]);
    try {
      execFileSync(process.execPath, ['--check', out], {stdio: 'pipe'});
      console.log(`  ok   ${path.basename(f)} script[${i}]  ${m[2].split('\n').length} lines`);
    } catch (e) {
      bad++;
      console.log(`  FAIL ${path.basename(f)} script[${i}]`);
      console.log(String(e.stderr || e.message).split('\n').slice(0, 6).join('\n'));
    }
    i++;
  }
  if (!i) console.log(`  --   ${path.basename(f)}: no inline script`);
}
process.exit(bad ? 1 : 0);
