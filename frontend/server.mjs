import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import {spawn} from 'node:child_process';
import {fileURLToPath} from 'node:url';
const root = path.dirname(fileURLToPath(import.meta.url));
const project = path.dirname(root);
const python = process.env.ATSA_PYTHON || path.join(project, '.venv', 'bin', 'python');
const api = spawn(python, [path.join(project, 'api.py')], {cwd:project, stdio:'inherit'});
api.on('error', error => {console.error('Unable to launch Python API:', error.message); process.exit(1);});
const types = {'.html':'text/html', '.js':'text/javascript', '.css':'text/css', '.svg':'image/svg+xml', '.png':'image/png', '.json':'application/json'};
const server = http.createServer((req,res) => {
  if (req.url.startsWith('/api/')) {
    const proxy = http.request({hostname:'127.0.0.1',port:8765,path:req.url,method:req.method}, response => {res.writeHead(response.statusCode, response.headers); response.pipe(res);});
    proxy.on('error', () => {res.writeHead(503, {'Content-Type':'application/json'});res.end(JSON.stringify({error:'Analysis service is starting. Please retry in a moment.'}));});
    req.pipe(proxy); return;
  }
  const urlPath = decodeURIComponent(new URL(req.url,'http://localhost').pathname);
  const file = path.resolve(root,'dist','.'+urlPath);
  const dist = path.join(root,'dist');
  if (!file.startsWith(dist+path.sep) && file !== dist) {res.writeHead(403);res.end();return;}
  const target = fs.existsSync(file) && fs.statSync(file).isFile() ? file : path.join(dist,'index.html');
  if (!fs.existsSync(target)) {res.writeHead(503);res.end('Build the frontend first: pnpm build');return;}
  res.writeHead(200,{'Content-Type':types[path.extname(target)] || 'application/octet-stream'});fs.createReadStream(target).pipe(res);
});
server.on('error', error => {api.kill(); console.error(error.message); process.exit(1);});
server.listen(Number(process.env.PORT || 3000),'127.0.0.1',()=>console.log('Forecast Studio: http://localhost:'+(process.env.PORT || 3000)));
process.on('exit',()=>api.kill());
const stop = () => {api.kill(); server.close(()=>process.exit(0));};
process.on('SIGINT',stop);process.on('SIGTERM',stop);
