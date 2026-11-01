// Reuse a healthy app, but report source changes instead of silently serving old code.
import {createHash} from 'node:crypto';
import {readFileSync,readdirSync} from 'node:fs';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
const project=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const sources=[path.join(project,'api.py'),...readdirSync(path.join(project,'src')).filter(f=>f.endsWith('.py')).sort().map(f=>path.join(project,'src',f))];
const hash=createHash('sha256');sources.forEach(f=>hash.update(readFileSync(f)));const revision=hash.digest('hex');
const url = `http://127.0.0.1:${process.env.PORT || 3000}`;
let health=null;
try {
 const response=await fetch(`${url}/api/health`,{signal:AbortSignal.timeout(1500)});
 if(response.ok)health=await response.json();
} catch { /* Start the server when no healthy app is listening. */ }
if(health?.status==='ok'){
 if(health.revision!==revision){
  console.log('Forecast Studio is running with older Python code.');
  console.log('Press Ctrl+C in its running terminal, then run ./run.sh again.');
 }else{
  console.log(`Forecast Studio is already running: ${url}`);
  console.log('Open that address in your browser.');
 }
}else{
 await import('./server.mjs');
}
