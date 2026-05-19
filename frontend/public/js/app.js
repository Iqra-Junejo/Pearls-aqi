"use strict";

/* ── AQI META ──────────────────────────────── */
const AMETA=[
  {max:50, l:"Good",                          c:"#22c55e",f:"😊",t:"Air quality is excellent. Enjoy outdoor activities freely."},
  {max:100,l:"Moderate",                      c:"#eab308",f:"😐",t:"Acceptable quality. Sensitive individuals should limit prolonged outdoor exertion."},
  {max:150,l:"Unhealthy for Sensitive Groups",c:"#f97316",f:"😷",t:"Sensitive groups should reduce outdoor activities."},
  {max:200,l:"Unhealthy",                     c:"#ef4444",f:"🤢",t:"Everyone may experience health effects. Limit outdoor exertion."},
  {max:300,l:"Very Unhealthy",                c:"#a855f7",f:"🤮",t:"Health alert. Avoid outdoor activities and keep windows closed."},
  {max:999,l:"Hazardous",                     c:"#be123c",f:"☠️",t:"Emergency. Stay indoors. Wear N95 mask if going out is unavoidable."},
];
const am=a=>AMETA.find(m=>a<=m.max)||AMETA[AMETA.length-1];

/* ── DEMO DATA ─────────────────────────────── */
const DEMO={
  current:{aqi:158,pm25:87.3,pm10:142.1,no2:38.7,o3:28.4,so2:12.1,co:0.9,
           temperature:34,humidity:68,wind_speed:14,pressure:1011},
  forecast:[
    {horizon:"24h",date:"Tomorrow",  weekday:"Tuesday",  aqi:178,category:"Unhealthy",                       color:"#ef4444",confidence:82},
    {horizon:"48h",date:"In 2 days", weekday:"Wednesday",aqi:118,category:"Unhealthy for Sensitive Groups",  color:"#f97316",confidence:76},
    {horizon:"72h",date:"In 3 days", weekday:"Thursday", aqi:94, category:"Moderate",                        color:"#eab308",confidence:70},
  ],
  metrics:{rmse:15.18,mae:11.90,r2:0.569,accuracy_15:81.4},
  shap:[
    {label:"PM2.5",    shap_value:42.1},{label:"Hour",     shap_value:18.3},
    {label:"AQI Lag",  shap_value:15.7},{label:"Humidity", shap_value:12.7},
    {label:"Wind Spd", shap_value:-11.2},{label:"Temp",   shap_value:-7.4},
    {label:"NO₂",      shap_value:6.8},{label:"AQI Roll",  shap_value:-5.3},
  ],
};
const POLLS=[
  {id:"PM2.5",key:"pm25",unit:"µg/m³",max:200,c:"#f97316"},
  {id:"PM10", key:"pm10",unit:"µg/m³",max:300,c:"#eab308"},
  {id:"NO₂",  key:"no2", unit:"µg/m³",max:200,c:"#8b5cf6"},
  {id:"O₃",   key:"o3",  unit:"µg/m³",max:200,c:"#3b82f6"},
  {id:"SO₂",  key:"so2", unit:"µg/m³",max:200,c:"#22c55e"},
  {id:"CO",   key:"co",  unit:"mg/m³", max:10, c:"#ec4899"},
];
const CITIES=[
  {flag:"🇵🇰",city:"Karachi",  meta:"34°C · PK",aqi:158},
  {flag:"🇮🇳",city:"New Delhi",meta:"29°C · IN",aqi:242},
  {flag:"🇦🇪",city:"Dubai",    meta:"38°C · AE",aqi:98},
  {flag:"🇬🇧",city:"London",   meta:"14°C · GB",aqi:32},
  {flag:"🇺🇸",city:"New York", meta:"18°C · US",aqi:67},
  {flag:"🇯🇵",city:"Tokyo",    meta:"20°C · JP",aqi:28},
];
const TIPS=[
  {i:"🏃",bg:"rgba(239,68,68,.1)",  h:"Avoid Outdoor Exercise",  t:"Prolonged outdoor activity increases pollutant inhalation. Choose indoor workouts today."},
  {i:"🪟",bg:"rgba(59,130,246,.1)", h:"Close Windows",           t:"Shut windows and doors to reduce indoor exposure from outdoor pollution sources."},
  {i:"😷",bg:"rgba(249,115,22,.1)", h:"Wear N95 Mask",           t:"If going out is unavoidable, N95 or KN95 masks significantly reduce PM2.5 exposure."},
  {i:"🌿",bg:"rgba(34,197,94,.1)",  h:"Use Air Purifier",        t:"Run a HEPA air purifier indoors and stay well hydrated."},
  {i:"🏥",bg:"rgba(168,85,247,.1)", h:"Monitor Symptoms",        t:"Watch for coughing, shortness of breath, or eye irritation. Consult a doctor."},
  {i:"📱",bg:"rgba(200,164,90,.1)", h:"Check Daily Forecast",    t:"Use Pearls AQI to plan outdoor activities on days when air quality improves."},
];

const $=id=>document.getElementById(id);
const set=(id,v)=>{const e=$(id);if(e)e.textContent=v};
const S={model:"lgbm",chart:null};

/* ════════════════════════════════
   LOADER — clip-path wipe out
════════════════════════════════ */
function initLoader(){
  const fill=$("ldrFill"),pct=$("ldrPct"),ldr=$("loader");
  let p=0;
  const iv=setInterval(()=>{
    p+=Math.random()*16+5;
    if(p>=100){
      p=100;clearInterval(iv);
      setTimeout(()=>{
        ldr.classList.add("out");
        setTimeout(()=>ldr.style.display="none",1000);
        startHeroAnims();
      },280);
    }
    fill.style.width=p+"%";
    pct.textContent=Math.round(p)+"%";
  },65);
}

/* ════════════════════════════════
   HERO ANIMATIONS — sequential
════════════════════════════════ */
function startHeroAnims(){
  // eyebrow line draws + text fades
  setTimeout(()=>{
    document.querySelector(".hero-eye-line")?.classList.add("in");
    document.querySelector(".hero-eye-txt")?.classList.add("in");
  },80);
  // words slide up one by one
  [".hero-w1",".hero-w2",".hero-w3",".hero-w4",".hero-w5"].forEach((sel,i)=>{
    setTimeout(()=>document.querySelector(sel)?.classList.add("in"),140+i*120);
  });
  // rule draws
  setTimeout(()=>$("heroRule")?.classList.add("in"),620);
  // strip
  setTimeout(()=>$("heroStrip")?.classList.add("in"),750);
  // foot
  setTimeout(()=>$("heroFoot")?.classList.add("in"),950);
}

/* ════════════════════════════════
   CUSTOM CURSOR
════════════════════════════════ */
function initCursor(){
  const c=$("cur"),f=$("curF");
  if(!c||!f||window.innerWidth<768)return;
  let mx=0,my=0,fx=0,fy=0;
  document.addEventListener("mousemove",e=>{
    mx=e.clientX;my=e.clientY;
    c.style.left=mx+"px";c.style.top=my+"px";
  },{passive:true});
  document.addEventListener("mousedown",()=>c.classList.add("big"));
  document.addEventListener("mouseup",()=>c.classList.remove("big"));
  document.querySelectorAll("a,button").forEach(el=>{
    el.addEventListener("mouseenter",()=>c.classList.add("gold"));
    el.addEventListener("mouseleave",()=>c.classList.remove("gold"));
  });
  (function loop(){fx+=(mx-fx)*.11;fy+=(my-fy)*.11;f.style.left=fx+"px";f.style.top=fy+"px";requestAnimationFrame(loop)})();
}

/* ════════════════════════════════
   NAVBAR
════════════════════════════════ */
function initNav(){
  const nav=document.querySelector(".nav");
  window.addEventListener("scroll",()=>nav.classList.toggle("on",scrollY>50),{passive:true});
  const tick=()=>{
    const t=new Date();
    set("navTime",t.toLocaleTimeString("en-PK",{hour:"2-digit",minute:"2-digit",hour12:false})+" PKT");
  };
  tick();setInterval(tick,1000);
}

/* ════════════════════════════════
   SPLIT WORDS — sec-h2
════════════════════════════════ */
function initSplitWords(){
  document.querySelectorAll("[data-split-words]").forEach(el=>{
    const words=el.textContent.trim().split(/\s+/);
    el.innerHTML=words.map((w,i)=>
      `<span class="sw"><span class="si" style="transition-delay:${i*.07}s">${w}</span></span>`
    ).join(" ");
  });
}

/* ════════════════════════════════
   INTERSECTION OBSERVER
════════════════════════════════ */
function initIO(){
  const io=new IntersectionObserver(entries=>{
    entries.forEach(e=>{
      if(!e.isIntersecting)return;
      e.target.classList.add("in");
      e.target.querySelectorAll(".fc-cfill[data-w]").forEach(b=>b.style.width=b.dataset.w+"%");
      io.unobserve(e.target);
    });
  },{threshold:.1,rootMargin:"0px 0px -36px 0px"});

  document.querySelectorAll(
    "[data-split-words],[data-reveal-tag],[data-rule],[data-clip],.fc-card,.wl-card,.tip-c"
  ).forEach(el=>io.observe(el));
}

/* ════════════════════════════════
   COUNT-UP (stats ticker)
════════════════════════════════ */
function initCountUp(){
  const io=new IntersectionObserver(entries=>{
    entries.forEach(e=>{
      if(!e.isIntersecting)return;
      const el=e.target,to=+el.dataset.to,t0=performance.now(),dur=1500;
      (function step(ts){
        const p=Math.min(1,(ts-t0)/dur),ease=1-Math.pow(1-p,3);
        el.textContent=Math.round(to*ease);
        if(p<1)requestAnimationFrame(step);
      })(t0);
      io.unobserve(el);
    });
  },{threshold:.3});
  document.querySelectorAll("[data-to]").forEach(el=>io.observe(el));
}

/* ════════════════════════════════
   COUNT-UP helper
════════════════════════════════ */
function countUp(el,to,ms=900){
  const t0=performance.now();
  (function s(ts){
    const p=Math.min(1,(ts-t0)/ms),e=1-Math.pow(1-p,3);
    el.textContent=Math.round(to*e);
    if(p<1)requestAnimationFrame(s);
  })(performance.now());
}

/* ════════════════════════════════
   RENDER FUNCTIONS
════════════════════════════════ */
function renderHero(c){
  const m=am(c.aqi);
  const el=$("hAqi");countUp(el,c.aqi);el.style.color=m.c;
  set("hStatus",m.l);
  set("hPm",`${c.pm25} µg/m³`);
  set("hTemp",`${c.temperature}°C`);
  set("hHum",`${c.humidity}%`);
  if(c.aqi>150){$("alertBand").style.display="block";set("alertTxt",`Air Quality Alert — AQI ${c.aqi}: ${m.l}. ${m.t}`);}
}

function renderGauge(c){
  const m=am(c.aqi),pct=Math.min(c.aqi/500,1);
  setTimeout(()=>{
    const arc=$("gArc");if(arc)arc.style.strokeDashoffset=257-pct*257;
    const nd=$("gNeedle");if(nd)nd.style.transform=`rotate(${-90+pct*180}deg)`;
  },350);
  const ne=$("gNum");countUp(ne,c.aqi);ne.style.color=m.c;
  set("gStatus",m.l);$("gStatus").style.color=m.c;
  set("gFace",m.f);
}

function renderEnv(c){
  set("eTemp",`${c.temperature}°`);
  set("eHum",`${c.humidity}%`);
  set("eWind",`${c.wind_speed}`);
  set("ePres",`${c.pressure}`);
}

function renderPolls(c){
  $("pollList").innerHTML=POLLS.map(p=>{
    const v=c[p.key]||0,pct=Math.min(100,(v/p.max)*100).toFixed(1);
    return`<div class="p-row">
      <span class="p-id" style="color:${p.c}">${p.id}</span>
      <div class="p-track"><div class="p-fill" style="background:${p.c}" data-w="${pct}"></div></div>
      <span class="p-val">${v} ${p.unit}</span>
    </div>`;
  }).join("");
  requestAnimationFrame(()=>document.querySelectorAll(".p-fill").forEach(b=>b.style.width=b.dataset.w+"%"));
}

function renderForecast(fc){
  $("fcGrid").innerHTML=fc.map((d,i)=>{
    const m=am(d.aqi);
    return`<div class="fc-card" style="--fcc:${m.c};transition-delay:${i*.1}s">
      <div class="fc-hor">${d.horizon} forecast</div>
      <div class="fc-date">${d.date}</div>
      <div class="fc-day">${d.weekday}</div>
      <div class="fc-aqi" style="color:${m.c}">${d.aqi}</div>
      <div class="fc-lbl" style="color:${m.c};background:${m.c}1a">${d.category}</div>
      <div class="fc-ct">Confidence: ${d.confidence}%</div>
      <div class="fc-cbar"><div class="fc-cfill" data-w="${d.confidence}"></div></div>
    </div>`;
  }).join("");
  setTimeout(()=>{
    const io=new IntersectionObserver(entries=>{entries.forEach(e=>{if(e.isIntersecting){e.target.classList.add("in");e.target.querySelectorAll(".fc-cfill").forEach(b=>b.style.width=b.dataset.w+"%");io.unobserve(e.target)}});},{threshold:.12});
    document.querySelectorAll(".fc-card").forEach(el=>io.observe(el));
  },50);
}

function renderShap(shap){
  const mx=Math.max(...shap.map(s=>Math.abs(s.shap_value)));
  $("shapBars").innerHTML=shap.map(s=>{
    const pos=s.shap_value>0,pct=(Math.abs(s.shap_value)/mx*100).toFixed(1),col=pos?"#ef4444":"#3b82f6";
    return`<div class="sh-r">
      <span class="sh-feat">${s.label}</span>
      <div class="sh-track"><div class="sh-fill ${pos?"p":"n"}" data-w="${pct}"></div></div>
      <span class="sh-num" style="color:${col}">${pos?"+":""}${s.shap_value.toFixed(1)}</span>
    </div>`;
  }).join("");
  requestAnimationFrame(()=>document.querySelectorAll(".sh-fill").forEach(b=>b.style.width=b.dataset.w+"%"));
}

function renderMets(m){
  if(!m)return;
  $("metsGrid").innerHTML=[
    {v:m.rmse?.toFixed(2)||"—",l:"RMSE"},
    {v:m.mae?.toFixed(2)||"—", l:"MAE"},
    {v:m.r2?.toFixed(3)||"—",  l:"R² Score"},
    {v:`${(m.accuracy_15||0).toFixed(1)}%`,l:"Accuracy ±15"},
  ].map(t=>`<div class="met-t"><div class="met-v">${t.v}</div><div class="met-l">${t.l}</div></div>`).join("");
}

function renderWatchlist(){
  $("wlGrid").innerHTML=CITIES.map((c,i)=>{
    const m=am(c.aqi);
    return`<div class="wl-card" style="transition-delay:${i*.07}s">
      <div class="wl-flag">${c.flag}</div>
      <div><div class="wl-city">${c.city}</div><div class="wl-meta">${c.meta}</div></div>
      <div class="wl-r">
        <div class="wl-num" style="color:${m.c}">${c.aqi}</div>
        <div class="wl-chip" style="color:${m.c};background:${m.c}1a">${m.l.split(" ")[0]}</div>
      </div>
    </div>`;
  }).join("");
  obsCards(".wl-card");
}

function renderTips(){
  $("tipsGrid").innerHTML=TIPS.map((t,i)=>
    `<div class="tip-c" style="transition-delay:${i*.07}s">
      <div class="tip-ico" style="background:${t.bg}">${t.i}</div>
      <div><div class="tip-title">${t.h}</div><div class="tip-txt">${t.t}</div></div>
    </div>`
  ).join("");
  obsCards(".tip-c");
}

function obsCards(sel){
  setTimeout(()=>{
    const io=new IntersectionObserver(entries=>{entries.forEach(e=>{if(e.isIntersecting){e.target.classList.add("in");io.unobserve(e.target)}});},{threshold:.08});
    document.querySelectorAll(sel).forEach(el=>io.observe(el));
  },60);
}

function buildHistory(){
  const d=[],now=Date.now();
  for(let h=23;h>=0;h--){
    const t=new Date(now-h*3600000);
    d.push({ts:t.getHours()+":00",aqi:125+Math.round(Math.sin(h*.45)*28+Math.random()*8)});
  }
  return d;
}

function renderTrend(hist){
  const vals=hist.map(h=>h.aqi),lbls=hist.map(h=>h.ts);
  set("tRange",`Min ${Math.min(...vals)} · Max ${Math.max(...vals)}`);
  if(S.chart)S.chart.destroy();
  S.chart=new Chart($("trendChart"),{
    type:"line",
    data:{labels:lbls,datasets:[{
      data:vals,borderColor:"rgba(240,234,214,.55)",borderWidth:1.6,
      pointBackgroundColor:vals.map(v=>am(v).c),pointRadius:3,pointHoverRadius:5,
      fill:true,
      backgroundColor:ctx=>{const g=ctx.chart.ctx.createLinearGradient(0,0,0,182);g.addColorStop(0,"rgba(240,234,214,.08)");g.addColorStop(1,"rgba(240,234,214,0)");return g},
      tension:.42
    }]},
    options:{
      responsive:true,maintainAspectRatio:false,
      plugins:{legend:{display:false},tooltip:{backgroundColor:"#1a1a16",borderColor:"rgba(240,234,214,.07)",borderWidth:1,titleColor:"#7a7468",bodyColor:"#f0ead6",callbacks:{label:c=>` AQI: ${c.parsed.y}`}}},
      scales:{
        x:{grid:{color:"rgba(240,234,214,.025)"},ticks:{color:"#3c3a34",font:{size:10,family:"DM Mono"},maxTicksLimit:8,autoSkip:true},border:{display:false}},
        y:{grid:{color:"rgba(240,234,214,.025)"},ticks:{color:"#3c3a34",font:{size:10,family:"DM Mono"}},border:{display:false},min:Math.max(0,Math.min(...vals)-18),max:Math.max(...vals)+18}
      }
    }
  });
}

function initModelSwitcher(){
  document.querySelectorAll(".mbtn").forEach(btn=>{
    btn.addEventListener("click",()=>{
      document.querySelectorAll(".mbtn").forEach(b=>b.classList.remove("on"));
      btn.classList.add("on");S.model=btn.dataset.model;
      const fc=DEMO.forecast.map(d=>({...d,aqi:d.aqi+Math.round((Math.random()-.5)*10),confidence:Math.max(60,d.confidence+Math.round((Math.random()-.5)*6))}));
      renderForecast(fc);
    });
  });
}

async function loadData(){
  let data=DEMO;
  try{const r=await fetch(`/api/predict?city=karachi&model=${S.model}`);if(r.ok)data=await r.json();}
  catch{/* demo */}
  renderHero(data.current);renderGauge(data.current);renderEnv(data.current);
  renderPolls(data.current);renderForecast(data.forecast);
  renderShap(data.shap||DEMO.shap);renderMets(data.metrics||DEMO.metrics);
  renderTrend(buildHistory());
  initIO();
}

/* ════════════════════════════════
   BOOT
════════════════════════════════ */
document.addEventListener("DOMContentLoaded",()=>{
  initLoader();
  initCursor();
  initNav();
  initSplitWords();
  initCountUp();
  renderWatchlist();
  renderTips();
  initModelSwitcher();
  loadData();
  setInterval(loadData,10*60*1000);
});
