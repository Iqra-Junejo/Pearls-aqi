/* particles.js — subtle cream particles for hero */
(function(){
  const c=document.getElementById("heroCanvas");if(!c)return;
  const x=c.getContext("2d");let W,H,pts=[],af;
  const C=["rgba(240,234,214,","rgba(200,164,90,","rgba(198,192,174,"];
  function sz(){W=c.width=c.offsetWidth;H=c.height=c.offsetHeight}
  function mk(){return{x:Math.random()*W,y:Math.random()*H,r:Math.random()*1.5+.3,vx:(Math.random()-.5)*.22,vy:(Math.random()-.5)*.18-.07,a:Math.random()*.2+.04,col:C[Math.floor(Math.random()*C.length)],p:Math.random()*Math.PI*2,ps:.007+Math.random()*.012}}
  function init(){sz();pts=Array.from({length:Math.floor(W*H/12000)},mk)}
  function draw(){
    x.clearRect(0,0,W,H);
    for(let i=0;i<pts.length;i++){
      for(let j=i+1;j<pts.length;j++){
        const a=pts[i],b=pts[j],dx=a.x-b.x,dy=a.y-b.y,d=Math.sqrt(dx*dx+dy*dy);
        if(d<110){x.beginPath();x.moveTo(a.x,a.y);x.lineTo(b.x,b.y);x.strokeStyle=`rgba(240,234,214,${(1-d/110)*.035})`;x.lineWidth=.5;x.stroke()}
      }
    }
    pts.forEach(p=>{
      p.p+=p.ps;const a=p.a*(.72+.28*Math.sin(p.p));
      x.beginPath();x.arc(p.x,p.y,p.r,0,Math.PI*2);
      x.fillStyle=p.col+a+")";x.fill();
      p.x+=p.vx;p.y+=p.vy;
      if(p.x<-8)p.x=W+8;if(p.x>W+8)p.x=-8;
      if(p.y<-8)p.y=H+8;if(p.y>H+8)p.y=-8;
    });
    af=requestAnimationFrame(draw);
  }
  window.addEventListener("resize",()=>{cancelAnimationFrame(af);init();draw()});
  init();draw();
})();
