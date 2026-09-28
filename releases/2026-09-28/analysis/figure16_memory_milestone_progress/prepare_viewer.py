"""Reuse the existing sphere viewer with milestone-based controls and annotations."""
from pathlib import Path
p=Path(__file__).resolve().parent
s=(p.parent/'figure15_memory_spherical_trajectories/viewer_template.html').read_text(encoding='utf-8')
def change(old,new):
 global s
 assert old in s,old[:80]
 s=s.replace(old,new)
change('记忆在球面上的时间轨迹','记忆与任务操作进度')
change('<label>速度 <select id="speed"><option value=".05">慢</option><option value=".1" selected>正常</option><option value=".2">快</option></select></label>', '<label>关键节点 <select id="milestone"></select></label><label><input type="checkbox" id="labels" checked>抓取标注</label>')
change('<span>时间</span>','<span>任务节点进度</span>')
change('<output id="timeLabel">1.000</output>','<output id="timeLabel">100%</output>')
change('#30123b,#466be3,#28bbec,#32f298,#a4fc3b,#eacb35,#fb7e21,#d02f05,#7a0403','#ccece6,#66c2a4,#238b45,#00441b')
change('<span>0 · 开始</span><span>归一化时间</span><span>1 · 结束</span>','<span>0%</span><span>任务节点进度 · 浅 → 深</span><span>100%</span>')
change('拖动旋转，滚轮缩放，悬停查看执行动作数。播放按各示范的归一化时间推进；大圆环标记最近一个真实采样点。起始时间没有点，是因为首次执行动作前的空记忆未纳入分析。','选择 G1、G2 等查看第几次抓取，P1、P2 等查看对应放置；播放按操作节点推进。五条示范使用各自的节点位置进行对齐。拖动旋转，滚轮缩放，悬停查看节点与百分比。圆环标记当前已显示的最后一个真实采样点。')
change('弧线是相邻采样点之间的球面插值，不是模型实际经过的中间记忆。球面归一化丢弃幅值；不同任务的坐标不可直接比较。','百分比是人工定义的等权操作节点完成比例，不是模型预测的进度。颜色不再依赖时间。G节点采用抬起参考帧，P节点采用放开动作后的边界；均经示范画面核对。标注附着在节点后第一个可用记忆采样点，可能有最多19步偏移。弧线仅为插值。')
change('d.time','d.progress')
change('time.toFixed(3)','`${Math.round(time*100)}%`')
change('document.getElementById(\'method\').textContent=`原1024维记忆 → 共同PCA三维 → 单位球面。T${task.task} 的前三个主成分保留 ${(100*task.variance).toFixed(1)}% 方差；五条示范使用同一个映射。`;', "document.getElementById('method').textContent=`每任务固定 ${task.demos[0].events.length} 个等权操作节点，百分比 = 已到达节点数 / 所需节点数。T4 Demo 2 的首次未抬起尝试、T10 Demo 5 的额外重抓不增加进度。原球面坐标保持不变。`;")
change("c.info.textContent=`${n} / ${d.points.length} 个记忆点${n?' · 动作 '+d.query[n-1]:''}`;", """if(document.getElementById('labels').checked){
 const labels=d.events.filter(ev=>ev.kind==='grasp'&&ev.point_index!==null&&ev.point_index<n).map(ev=>({ev,p:project(d.points[ev.point_index],c,w,h)}));
 for(const side of [-1,1]){const group=labels.filter(v=>(v.p[0]<w/2?-1:1)===side).sort((a,b)=>a.p[1]-b.p[1]);
 group.forEach((v,j)=>{const x=side<0?w*.07:w*.93,y=group.length===1?h*.5:h*(.2+.6*j/(group.length-1));ctx.strokeStyle='#648b78';ctx.lineWidth=.65;ctx.beginPath();ctx.moveTo(v.p[0],v.p[1]);ctx.lineTo(x,y);ctx.stroke();ctx.font='13px Arial';ctx.textAlign=side<0?'right':'left';ctx.fillStyle='#16452c';ctx.fillText(v.ev.label,x+(side<0?-2:2),y+4)})}ctx.textAlign='left';}
 c.info.textContent=n?`${Math.round(d.progress[n-1]*100)}% · ${d.stages[n-1]}`:'暂无该节点后的记忆采样点';""")
change('执行动作数：${d.query[hit.i]}\\n归一化时间：${d.progress[hit.i].toFixed(3)}','${d.stages[hit.i]}\\n节点进度：${Math.round(d.progress[hit.i]*100)}%\\n记忆采样：动作 ${d.query[hit.i]}')
old="function tick(now){if(!running)return;let t=Math.min(1,+slider.value+Math.min((now-last)/1000,.1)*+document.getElementById('speed').value);last=now;slider.value=t;draw();if(t>=1)stop();else requestAnimationFrame(tick)}"
new="function tick(now){if(!running)return;const total=DATA[+sel.value].demos[0].events.length;if(now-last>=1100){last=now;const index=Math.round(+slider.value*total)+1;slider.value=Math.min(1,index/total);document.getElementById('milestone').value=Math.min(index,total);draw();if(index>=total){stop();return}}requestAnimationFrame(tick)}"
change(old,new)
change("sel.onchange=()=>{stop();slider.value=1;draw()}","sel.onchange=()=>{stop();slider.value=1;fillMilestones();draw()}")
change("slider.oninput=()=>{stop();draw()}","slider.oninput=()=>{stop();document.getElementById('milestone').value=Math.round(+slider.value*DATA[+sel.value].demos[0].events.length);draw()}")
change("window.addEventListener('resize',draw);draw();", """function fillMilestones(){const menu=document.getElementById('milestone');menu.replaceChildren(new Option('初始接近 · 0%',0));const evs=DATA[+sel.value].demos[0].events;evs.forEach((ev,j)=>menu.add(new Option(`${ev.label} · ${ev.description} · ${Math.round(ev.progress*100)}%`,j+1)));menu.value=evs.length;slider.step=1/evs.length;}
document.getElementById('milestone').onchange=()=>{stop();slider.value=+document.getElementById('milestone').value/DATA[+sel.value].demos[0].events.length;draw()};document.getElementById('labels').onchange=draw;window.addEventListener('resize',draw);fillMilestones();draw();""")
change('ctx.globalAlpha=p[2]<0?.55:1','ctx.globalAlpha=1')
# Continuous slider avoids browser snapping inaccuracies for fractions such as 1/14.
s=s.replace('slider.step=1/evs.length;','slider.step="any";')
(p/'viewer_template.html').write_text(s,encoding='utf-8')
print('Milestone viewer template ready')
