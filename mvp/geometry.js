/* All positions and ranges use meters. X is forward, Y is left, Z is up. */
(function (root) {
  const STEP = 0.2, SIZE = 80, MIN = -8;
  const Z = [0.04, 0.35, 0.85, 1.65, 2.65];
  const UNKNOWN = 0, FREE = 1, OCCUPIED = 2, SHADOW = 3;
  const W = 128, H = 80, CX = W / 2, CY = H / 2;
  const FOCAL = (H / 2 - 1) / (95 * Math.PI / 180);
  const RANGE = 21;
  const COLORS = {
    sky: [102, 135, 142], ground: [52, 62, 64], ego: [64, 142, 129],
    rider: [242, 166, 91], pedestrian: [241, 203, 124], car: [112, 149, 180],
    wall: [112, 126, 122], curb: [154, 152, 130], stone: [213, 151, 96],
    plant: [102, 145, 86], truck: [147, 165, 170], glass: [42, 54, 64], follower: [128,161,189]
  };
  const camera = (name, position, yaw, pitch) => ({
    name, position, yaw, pitch, right: [Math.sin(yaw), -Math.cos(yaw), 0],
    forward: [Math.cos(yaw) * Math.cos(pitch), Math.sin(yaw) * Math.cos(pitch), Math.sin(pitch)],
    up: [-Math.cos(yaw) * Math.sin(pitch), -Math.sin(yaw) * Math.sin(pitch), Math.cos(pitch)]
  });
  const CAMS = [
    camera("Trước", [2.13, 0, 0.83], 0, -0.23),
    camera("Sau", [-2.13, 0, 0.84], Math.PI, -0.23),
    camera("Trái", [0.59, 1.04, 1.14], Math.PI / 2, -0.33),
    camera("Phải", [0.59, -1.04, 1.14], -Math.PI / 2, -0.33)
  ];
  const CODES = Object.keys(COLORS);
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const clamp = (x, min, max) => Math.max(min, Math.min(max, x));

  function rayAt(cam, u, v) {
    const x = u - CX, y = CY - v;
    const r = Math.hypot(x, y);
    const theta = r / FOCAL;
    if (theta > 95 * Math.PI / 180) return null;
    const k = r > 0 ? Math.sin(theta) / r : 0;
    return cam.forward.map((a, i) => a * Math.cos(theta) + k * (x * cam.right[i] + y * cam.up[i]));
  }

  function project(cam, point) {
    const v = point.map((x, i) => x - cam.position[i]);
    const right = dot(v, cam.right), up = dot(v, cam.up);
    const plane = Math.hypot(right, up);
    const a = Math.atan2(plane, dot(v, cam.forward));
    if (a > 95 * Math.PI / 180 || Math.hypot(...v) < 1e-8) return null;
    const s = plane > 1e-9 ? a * FOCAL / plane : 0;
    return [CX + s * right, CY - s * up];
  }

  const rayTables = CAMS.map(cam => Array.from({length: W * H}, (_, i) => rayAt(cam, i % W + 0.5, Math.floor(i / W) + 0.5)));

  function box(x, y, l, w, h, kind, z = 0, id = "") {
    return {min: [x - l / 2, y - w / 2, z], max: [x + l / 2, y + w / 2, z + h], kind, id};
  }

  function futurePose(v, yaw, t) {
    const a = yaw * t;
    if (Math.abs(yaw) < 1e-8) return {x: v * t, y: 0, yaw: a};
    return {x: v * Math.sin(a) / yaw, y: v * (1 - Math.cos(a)) / yaw, yaw: a};
  }

  function toLocal(p, pose) {
    const x = p[0] - pose.x, y = p[1] - pose.y;
    const c = Math.cos(pose.yaw), s = Math.sin(pose.yaw);
    return [c*x + s*y, -s*x + c*y, p[2]];
  }

  function toWorld(p, pose) {
    const c = Math.cos(pose.yaw), s = Math.sin(pose.yaw);
    return [pose.x + c*p[0] - s*p[1], pose.y + s*p[0] + c*p[1], p[2]];
  }

  function cast(origin, ray, objects) {
    let best = ray[2] < -1e-8 ? Math.min(RANGE, -origin[2] / ray[2]) : RANGE;
    let hit = best < RANGE ? {kind: "ground"} : {kind: "sky"};
    let normal = 2;
    for (const obj of objects) {
      let start = 0.025, end = best, axis = 2;
      for (let i = 0; i < 3; i++) {
        if (Math.abs(ray[i]) < 1e-12) {
          if (origin[i] < obj.min[i] || origin[i] > obj.max[i]) { end = -1; break; }
          continue;
        }
        const a = (obj.min[i] - origin[i]) / ray[i], b = (obj.max[i] - origin[i]) / ray[i];
        const enter = Math.min(a,b);
        if (enter > start) { start = enter; axis = i; }
        end = Math.min(end, Math.max(a,b));
        if (end < start) break;
      }
      if (end >= start && start < best) { best = start; hit = obj; normal = axis; }
    }
    return {range: best, kind: hit.kind, id: hit.id, normal};
  }

  function capture(pose, objects, health) {
    const car = [box(0,0,4.2,1.88,0.63,"ego",0.32), box(-0.25,0,2.1,1.66,0.55,"ego",0.95)];
    return CAMS.map((cam, c) => {
      const ranges = new Float32Array(W*H);
      const kinds = new Uint8Array(W*H);
      const image = new Uint8ClampedArray(W*H*4);
      const origin = toWorld(cam.position,pose);
      const cs = Math.cos(pose.yaw), sn = Math.sin(pose.yaw);
      const blocked = health[c].blocked;
      for (let i = 0; i < W*H; i++) {
        const r = rayTables[c][i];
        const offset = i*4;
        image[offset+3] = 255;
        if (!r) { image[offset] = 12; image[offset+1] = 17; image[offset+2] = 16; continue; }
        const dir = [r[0]*cs-r[1]*sn, r[0]*sn+r[1]*cs, r[2]];
        let hit = cast(origin, dir, objects);
        const ego = cast(cam.position, r, car);
        if (ego.kind === "ego" && ego.range < hit.range) hit = ego;
        ranges[i] = hit.range;
        kinds[i] = CODES.indexOf(hit.kind);
        const p = origin.map((a,j) => a + dir[j]*hit.range);
        let color = COLORS[hit.kind];
        if (hit.kind === "ground") {
          const dash = Math.abs(p[1]) < 0.045 && ((p[0] + 30) % 3) < 1.6;
          const side = Math.abs(Math.abs(p[1])-3.7) < 0.055;
          if (dash || side) color = [191, 181, 140];
          if (Math.abs(p[1]) > 4.05) color = [78,87,85];
        }
        const glareMask = blocked > 0 && ((i % W - 60) ** 2 / 1720 + (Math.floor(i / W) - 47) ** 2 / 740 < blocked * 2.9);
        const shade = hit.kind === "sky" ? 1 + r[2]*0.32 : 1.08 - hit.normal*0.055 - hit.range*0.008;
        for (let k = 0; k < 3; k++) {
          const noise = ((i * 31 + c * 13) % 11 - 5)*0.55;
          image[offset+k] = health[c].offline ? 14 : glareMask ? (health[c].type === "glare" ? 245 : [69,52,39][k]) : color[k]*shade + noise;
        }
        if (glareMask) ranges[i] = 0;
      }
      return {ranges, kinds, image};
    });
  }

  function indices(p) {
    const x = Math.floor((p[0]-MIN)/STEP), y = Math.floor((p[1]-MIN)/STEP);
    let z = -1;
    for (let i=0;i<Z.length-1;i++) if (p[2]>=Z[i] && p[2]<Z[i+1]) { z=i; break; }
    return x<0 || x>=SIZE || y<0 || y>=SIZE || z<0 ? -1 : x+SIZE*(y+SIZE*z);
  }

  function center(index) {
    const x = index % SIZE, y = Math.floor(index / SIZE) % SIZE, z = Math.floor(index / (SIZE*SIZE));
    return [MIN+(x+0.5)*STEP, MIN+(y+0.5)*STEP, (Z[z]+Z[z+1])/2];
  }

  function traverse(origin, ray, stop, visit, start = 0) {
    const point = t => origin.map((a,i) => a + ray[i]*t);
    let t = start + 1e-7;
    while (t < stop) {
      const p = point(t);
      const x = Math.floor((p[0]-MIN)/STEP), y = Math.floor((p[1]-MIN)/STEP);
      let z = Z.findIndex(e => e > p[2]) - 1;
      if (p[2]>=Z[Z.length-1]) z = Z.length-1;
      if (x<0 || x>=SIZE || y<0 || y>=SIZE || z<0 || z>=Z.length-1) break;
      visit(x+SIZE*(y+SIZE*z));
      const edge = [
        MIN + (x+(ray[0]>0?1:0))*STEP, MIN + (y+(ray[1]>0?1:0))*STEP,
        Z[z+(ray[2]>0?1:0)]
      ];
      let next = stop;
      for(let i=0;i<3;i++) {
        if (Math.abs(ray[i]) < 1e-12) continue;
        const at = (edge[i]-origin[i])/ray[i];
        if(at > t-1e-8) next = Math.min(at,next);
      }
      t = Math.max(t+1e-7,next+1e-7);
    }
  }

  function reconstruct(images, health) {
    const length = SIZE*SIZE*(Z.length-1);
    const cells = new Uint8Array(length), hitKind = new Uint8Array(length), support = new Uint8Array(length);
    const detections = new Map(), lowPoints = new Map();
    let rays = 0, points = 0, reprojection = 0;
    for(let c=0;c<CAMS.length;c++) {
      if (health[c].gated) continue;
      const cam = CAMS[c], input = images[c];
      for (let y=1;y<H;y+=2) for (let x=1;x<W;x+=2) {
        const i = x+y*W, ray = rayTables[c][i], t = input.ranges[i];
        const kind = CODES[input.kinds[i]];
        if (!ray || !t || kind === "ego") continue;
        rays++;
        const p = cam.position.map((a,j) => a + ray[j]*(t+0.0001));
        const occupied = kind !== "ground" && kind !== "sky";
        const index = occupied ? indices(p) : -1;
        traverse(cam.position,ray,t,(j)=>{
          if(cells[j]!==OCCUPIED) cells[j]=FREE;
          support[j]|=1<<c;
        });
        if (index<0) continue;
        cells[index]=OCCUPIED;
        hitKind[index]=input.kinds[i];
        support[index]|=1<<c;
        points++;
        const uv = project(cam,p);
        if (uv) reprojection = Math.max(reprojection,Math.hypot(uv[0]-x-0.5,uv[1]-y-0.5));
        traverse(cam.position,ray,t+4,(j)=>{ if(cells[j]===UNKNOWN) cells[j]=SHADOW; },t+0.004);
        if (["rider","follower","pedestrian"].includes(kind)) {
          const label = kind==="follower"?"car":kind;
          const sum = detections.get(label) || {pos:[0,0,0], n:0};
          sum.pos = sum.pos.map((a,k)=>a+p[k]); sum.n++;
          detections.set(label,sum);
        }
        if (kind==="stone" && p[2] > 0.12 && p[2] < 0.48) {
          const key=p.slice(0,2).map(a=>Math.floor(a/0.05)).join(",");
          const prev=lowPoints.get(key);
          if(!prev || p[2]>prev[2]) lowPoints.set(key,p);
        }
      }
    }
    const counts = [0,0,0,0], occupied = [];
    cells.forEach((v,i)=>{counts[v]++; if(v===OCCUPIED) occupied.push(i);});
    return {cells, hitKind, support, counts, occupied, rays, points, reprojection,
      detections: Array.from(detections,([kind,sum])=>({kind, pos:sum.pos.map(x=>x/sum.n), hits:sum.n})),
      lowPoints: [...lowPoints.values()]};
  }

  function surfaceError(occupied, objects, pose) {
    if(!occupied.length)return null;
    let sum=0,max=0;
    for(const index of occupied) {
      const p=toWorld(center(index),pose);
      let nearest=Infinity;
      for(const obj of objects){
        const d=p.map((value,i)=>Math.max(obj.min[i]-value,value-obj.max[i]));
        const outside=Math.hypot(...d.map(value=>Math.max(0,value)));
        const distance=outside || -Math.max(...d);
        nearest=Math.min(nearest,distance);
      }
      sum+=nearest;max=Math.max(max,nearest);
    }
    return {mean:sum/occupied.length,max};
  }

  const api = {STEP,SIZE,MIN,Z,UNKNOWN,FREE,OCCUPIED,SHADOW,W,H,CAMS,CODES,COLORS,
    clamp,rayAt,project,cast,box,futurePose,toLocal,toWorld,indices,center,capture,reconstruct,surfaceError};
  if (typeof module !== "undefined") module.exports = api;
  else root.Geometry = api;
})(globalThis);
