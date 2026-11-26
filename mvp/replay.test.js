const test = require("node:test");
const assert = require("node:assert/strict");
const G = require("./geometry");
const R = require("./replay");

test("Tia fisheye chiếu vòng ở tâm, biên và ngoài 90 độ", () => {
  for (const camera of G.CAMS) {
    for (const [u, v] of [[64,40],[81,54],[25.5,40],[64,78.5]]) {
      const ray=G.rayAt(camera,u,v);
      const p=camera.position.map((x,i)=>x+2.8*ray[i]);
      const [uu,vv]=G.project(camera,p);
      assert.ok(Math.hypot(uu-u,vv-v)<1e-8);
      assert.ok(Math.abs(Math.hypot(...ray)-1)<1e-12);
    }
    assert.equal(G.rayAt(camera,0,0),null);
  }
});

test("Khối chuẩn 1 m cho độ sâu đúng, voxel có vật, trước mặt trống và sau mặt khuất", () => {
  const obstacle=G.box(4.5,0,1,1,1,"wall");
  const centerline=G.cast([2.13,0,0.65],[1,0,0],[obstacle]);
  assert.ok(Math.abs(centerline.range-1.87)<1e-10);
  const pose={x:0,y:0,yaw:0};
  const health=G.CAMS.map((_,i)=>({gated:i!==0,blocked:0}));
  const scan=G.reconstruct(G.capture(pose,[obstacle],health),health);
  assert.equal(scan.cells[G.indices([3.4,0.1,0.6])],G.FREE);
  assert.equal(scan.cells[G.indices([4.1,0.1,0.6])],G.OCCUPIED);
  assert.equal(scan.cells[G.indices([4.7,0.1,0.6])],G.SHADOW);
  const error=G.surfaceError(scan.occupied,[obstacle],pose);
  // The top surface shares a 0.8 m vertical band whose center is at 1.25 m.
  assert.ok(error.mean<0.15);
  assert.ok(error.max<=Math.hypot(G.STEP,G.STEP,1)/2+1e-4);
});

test("F01–F12 có tình huống kích hoạt từ dữ liệu mô phỏng", () => {
  const activated=new Set();
  for(const scene of R.SCENES) {
    const replay=new R.Replay(scene.key);
    if(scene.key==="blind")replay.fault(1,"offline",0.25);
    const frame=replay.get(scene.start/R.DT);
    for(const id of scene.features) {
      assert.ok(frame.features[id].active,`${scene.key}: ${id} ${frame.features[id].summary}`);
      activated.add(id);
    }
    if(scene.key==="under"){
      assert.ok(frame.features.F06.under.some(p=>p.seen<frame.t-0.5),
        "Điểm dưới gầm phải tồn tại qua nhiều frame");
    }
    if(scene.key==="slot"){
      assert.ok(frame.features.F10.slot.length>=5);
    }
  }
  assert.equal(activated.size,12);
});

test("Camera bị chặn giữ lịch sử trạng thái và bốn camera mất tín hiệu cho unknown", () => {
  const replay=new R.Replay("blind");
  replay.fault(0,"glare",0);
  assert.equal(replay.get(2).health[0].gated,false);
  assert.equal(replay.get(8).health[0].gated,true);
  replay.fault(0,"ok",1.25);
  assert.equal(replay.get(8).health[0].type,"glare");
  assert.equal(replay.get(24).health[0].gated,false);
  for(let i=0;i<4;i++)replay.fault(i,"offline",3.5);
  const frame=replay.get(28);
  assert.deepEqual(frame.scan.counts,[G.SIZE*G.SIZE*4,0,0,0]);
  assert.equal(frame.scan.rays,0);
  assert.ok(frame.features.F11.active);
});

test("Hộp đen lưu 5 giây và duy trì đúng số voxel mỗi frame", () => {
  const replay=new R.Replay("mixed");
  const archive=replay.clip(16);
  assert.equal(archive.frames.length,41);
  assert.equal(archive.frames.at(-1).t-archive.frames[0].t,5);
  assert.ok(archive.score>=0 && archive.score<=100);
  assert.ok(archive.frames.some(f=>f.ttc!==null && f.ttc<1.2));
  for(const f of archive.frames){
    const length=f.state.reduce((sum,value,i)=>sum+(i%2?value:0),0);
    assert.equal(length,G.SIZE*G.SIZE*4);
    assert.deepEqual(f.pose,replay.get(f.t/R.DT).pose);
  }
});
