const { chromium } = require('playwright'); const fs=require('fs');
(async()=>{
 const b=await chromium.launch({args:['--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader']});
 const p=await b.newPage({viewport:{width:1920,height:1080}});
 p.on('console',m=>console.log('C',m.text())); p.on('pageerror',e=>console.log('E',e.message));
 await p.goto('http://127.0.0.1:8799/gt_mocap.html'); await p.waitForFunction('window.ready',null,{timeout:120000});
 const n=await p.evaluate('window.nFrames'); const gt=[]; const el=await p.$('#c');
 fs.mkdirSync('mocap/frames',{recursive:true});
 const t0=Date.now();
 for(let i=0;i<n;i++){ gt.push(await p.evaluate(i=>window.renderFrame(i),i));
  await el.screenshot({path:`mocap/frames/f_${String(i).padStart(5,'0')}.jpg`,type:'jpeg',quality:95});
  if(i%100==0) console.log(i,(Date.now()-t0)/1000); }
 fs.writeFileSync('mocap/gt.json',JSON.stringify(gt)); await b.close();
})();
