import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

let project = null;
let meta = null;
let activeView = '2d';
let autoSpin = true;
let wireframeOn = false;
let lang = localStorage.getItem('pf_lang') || ((navigator.language || '').toLowerCase().startsWith('zh') ? 'zh' : 'en');

const I18N = {
  zh: {
    brand_tag: '线盘神灯',
    brand_sub: '线盘神灯 · 对话许愿 → 2D 确认 → 真 3D 与拓竹 AMS / 免胶水 .3MF',
    step1: '参考与对话', step2: '2D 确认颜色尺寸', step3: '3D 预览与拆件', step4: '导出 .3MF 打印',
    new_btn: '+ 新建', mobile_btn: '📱 手机', inspector_btn: '⚙️ 参数', lang_btn: 'EN',
    mnav_chat: '对话', mnav_canvas: '2D / 3D', mnav_mw: '搜索', mnav_specs: '参数导出',
    modal_mobile_title: '📱 手机 App / 同 Wi-Fi 扫码直连',
    copy_btn: '复制', copied_btn: '已复制',
    lan_ok: '✅ 局域网已开放，手机连同一 Wi-Fi 扫码即可使用',
    lan_local_only: '⚠️ 当前仅本机监听 (127.0.0.1)，请加 --lan 启动以开放手机扫码连接',
    lan_hint_ok: (ip) => `手机与当前电脑连同一局域网 Wi-Fi（本机 IP: ${ip}），手机端可直接对话生成 2D/3D 并下载 .3MF，算力自动走当前电脑 RTX GPU / Apple Silicon。`,
    lan_hint_local: (cmd) => `在终端运行「${cmd}」或使用桌面版 App（默认开启局域网连接），再扫码即可手机直连。`,
    guide_ios_t: '🍎 iOS (iPhone / iPad):',
    guide_ios_d: 'Safari 扫码打开 → 点底部分享图标 →「添加到主屏幕」，全屏 App 运行（也可用仓库 mobile/ 编译原生 Capacitor App）。',
    guide_and_t: '🤖 Android:',
    guide_and_d: 'Chrome 扫码打开 → 点右上菜单「安装应用 / 添加到主屏幕」，或用仓库 mobile/ 打包原生 APK。',
    inline_2d_label: '2D 效果图', inline_zoom: '放大看图', inline_build3d: '⚡ 生成 3D',
    tab_chat: '💬 对话设计', tab_mw: '🔎 MakerWorld 搜索',
    quick_label: '快捷示例：', attach_ref: '📎 参考图/STL', auto_2d: '自动画2D', send_btn: '发送 ↵',
    chat_ph: '输入想法、尺寸(mm)、配色、AMS台数或免胶水要求… (Enter 发送，支持 Ctrl+V 贴图)',
    mw_note: '直接搜索 MakerWorld 公开模型并一键引用，或粘贴模型链接 / Ctrl+V 粘贴截图提取结构与色盘。',
    mw_ph: '搜索 MakerWorld（如 wall-e, gridfinity, headphone stand…）',
    search_btn: '搜索', mw_url_label: '粘贴指定 MakerWorld 模型链接', import_btn: '导入',
    use_ref_btn: '+ 引用参考', open_mw_btn: '打开',
    thinking_msg: '正在思考设计方案并更新规格…',
    rendering_2d_msg: '规格已更新，正在后台渲染 2D 确认图…',
    attached_label: '当前已绑定参考素材', no_refs: '暂无绑定参考', refs_count: (n) => `${n} 参考`,
    view_2d: '🎨 2D 概念图', view_3d: '🧊 3D 交互模型', view_split: '🧩 无 AMS 分块排盘', view_tt: '📐 四视角验收图',
    confirm_btn: '⚡ 确认方案并生成 3D 打印文件',
    empty_2d_title: '从一句话开始，直接出可打印的 .3MF',
    empty_2d_desc: '左侧描述你想做的角色或功能件（或点击顶部切换已生成的瓦力笔筒 / 无AMS耳机架），先确认 2D 效果图与颜色尺寸，再一键生成水密 3D 模型。',
    empty_tt_title: '四视角 3D 验收图', empty_tt_desc: '点击右上角确认并生成 3D 后，这里会自动展示 3/4、正面、侧面、背面四视角成品实色图。',
    cam_iso: '3/4 视角', cam_front: '正面', cam_side: '侧面', cam_top: '俯视', cam_wire: '线框', cam_spin: '旋转',
    card_hw: '🖨️ 打印机与 AMS 颜色预算', printer_label: '目标机型', ams_label: '你的 AMS 台数（决定最终颜色取舍上限）',
    ams0_t: '0 台', ams0_s: '无AMS', ams1_t: '1 台', ams1_s: '≤4色', ams2_t: '2 台', ams2_s: '≤8色', ams3_t: '3 台', ams3_s: '≤12色', ams4_t: '4 台', ams4_s: '≤16色',
    mode_label: '多色策略', mode_auto: '自动匹配', mode_ams: 'AMS 同盘换色', mode_split: '无AMS分块 (0废料)', mode_mono: '合并单色',
    joint_label: '免胶水机械连接', joint_dt: '燕尾槽 (最稳抗拔)', joint_sp: '弹性开槽卡扣销', joint_pt: '紧配榫卯', joint_none: '不加连接器',
    card_dims: '📏 尺寸 (mm) 与耗材色盘', add_color: '+ 加一色', dim_w: '宽 W', dim_d: '深 D', dim_h: '高 H',
    swatch_label: '拓竹 PLA 常用色板（点击添加到最后槽位）', apply_spec: '应用修改并刷新 2D 预览',
    card_dfm: '🛡️ 打印体检 (DFM) 与颜色取舍', m_mass: '耗材', m_flush: '冲刷废料', m_time: '预估耗时',
    dfm_empty: '确认 2D 方案并点击生成后显示完整水密体检、耗时与 AMS 颜色取舍日志。',
    card_export: '📦 导出 3D 打印文件', export_empty: '等待确认方案并生成…',
    slot_lbl: (s) => `槽 ${s}`, cap_no_ams: '无 AMS (免胶水拆件/单色)', cap_ams: (n) => `≤ ${n * 4} 色上限`,
    welcome: '你好！直接告诉我你想打印的任何角色、动植物或实用功能件、尺寸(mm)和 AMS 台数；也可以切到上方「MakerWorld 搜索」找参考，或直接 Ctrl+V 贴图。',
    watertight_ok: '✅ 水密可打印', watertight_warn: '⚠️ 需切片修补',
    hud_mode: (d) => `${d.effective_multicolor_mode} · ${d.ams_slots_used}色 · 接头:${d.joint_type_used}`,
    status_3d_ready: '3D 模型已就绪，可鼠标左键旋转、滚轮缩放',
    status_loaded: (title) => `已载入项目：${title}`,
    dl_3mf: '⬇ 拓竹多色 .3MF', dl_3mf_tag: '推荐直拖切片',
    dl_split: '⬇ 无AMS免胶水排盘 .3MF', dl_split_tag: '0废料',
    dl_stl: '⬇ 通用单网格 .STL', dl_stl_tag: '全切片器',
    dl_step: '⬇ 工程参数化 .STEP', dl_step_tag: 'CAD编辑',
    dl_py: '⬇ build123d 源码 .PY', dl_gcode: '⬇ 已切片 .GCODE.3MF', dl_part: (k) => `⬇ 分色零件 ${k}.stl`,
    chips: [
      { label: '🤖 瓦力功能创意', prompt: '我想生成一个机器人总动员里面的瓦力，但是又不想完全弄成手办想有一些功能，你有啥建议' },
      { label: '🎧 无AMS燕尾槽耳机架', prompt: '做一个桌面耳机支架，带底部理线槽，高210mm，哑光黑+橘红，没有AMS，帮我用燕尾槽免胶水拆件打印' },
      { label: '🦾 钢铁侠无线充座', prompt: '做一个漫威钢铁侠Q版战甲胸口MagSafe无线充底座，高125mm，1台AMS四色打印' },
      { label: '📦 Gridfinity收纳盒', prompt: '做一个Gridfinity风格3分区桌面收纳盒，150x100x85mm，1台AMS三色' },
    ],
  },
  en: {
    brand_tag: 'Wish → 3MF',
    brand_sub: 'Wish on a spool → 2D confirm → True 3D & Bambu AMS / Glue-Free .3MF',
    step1: 'Refs & Chat', step2: '2D Color & Size', step3: '3D & Split Preview', step4: 'Export .3MF',
    new_btn: '+ New', mobile_btn: '📱 Mobile', inspector_btn: '⚙️ Specs', lang_btn: '中文',
    mnav_chat: 'Chat', mnav_canvas: '2D / 3D', mnav_mw: 'Search', mnav_specs: 'Specs',
    modal_mobile_title: '📱 Mobile App / Same-Wi-Fi QR Pairing',
    copy_btn: 'Copy', copied_btn: 'Copied',
    lan_ok: '✅ LAN Mode Active — scan on iPhone or Android on the same Wi-Fi',
    lan_local_only: '⚠️ Localhost-only mode (127.0.0.1) — start with --lan to enable phone scanning',
    lan_hint_ok: (ip) => `Connect your phone to the same Wi-Fi (host IP: ${ip}). All heavy SDXL / Hunyuan3D / CAD computation runs on this PC/Mac GPU automatically.`,
    lan_hint_local: (cmd) => `Run "${cmd}" (or launch the Desktop App which enables LAN by default) to connect from your phone.`,
    guide_ios_t: '🍎 iOS (iPhone / iPad):',
    guide_ios_d: 'Open in Safari → Share → "Add to Home Screen" for full-screen PWA app (or build native iOS via mobile/ Capacitor project).',
    guide_and_t: '🤖 Android:',
    guide_and_d: 'Open in Chrome → Menu → "Install app / Add to Home Screen", or build a native APK from mobile/.',
    inline_2d_label: '2D Concept', inline_zoom: 'View Full', inline_build3d: '⚡ Build 3D',
    tab_chat: '💬 Designer Chat', tab_mw: '🔎 MakerWorld Search',
    quick_label: 'Quick prompts:', attach_ref: '📎 Ref / STL', auto_2d: 'Auto 2D', send_btn: 'Send ↵',
    chat_ph: 'Describe what to print, dimensions (mm), colors, AMS count… (Enter to send, Ctrl+V image)',
    mw_note: 'Search live MakerWorld models and attach any as reference, or paste a URL / Ctrl+V a screenshot.',
    mw_ph: 'Search MakerWorld (e.g. wall-e, gridfinity, headphone stand…)',
    search_btn: 'Search', mw_url_label: 'Paste MakerWorld Model URL', import_btn: 'Import',
    use_ref_btn: '+ Use Ref', open_mw_btn: 'Open',
    thinking_msg: 'Designing proposal & updating specs…',
    rendering_2d_msg: 'Spec updated — rendering 2D concept preview in background…',
    attached_label: 'Attached References', no_refs: 'No references attached', refs_count: (n) => `${n} ref`,
    view_2d: '🎨 2D Concept', view_3d: '🧊 Interactive 3D', view_split: '🧩 No-AMS Split Plates', view_tt: '📐 4-Angle Sheet',
    confirm_btn: '⚡ Confirm & Build 3D Print Files',
    empty_2d_title: 'Chat Your Idea → Ready-to-Slice .3MF',
    empty_2d_desc: 'Describe a character hybrid or custom functional part on the left (or switch to the pre-built WALL-E Pen Holder / No-AMS Headphone Stand above).',
    empty_tt_title: '4-Angle 3D Verification Sheet', empty_tt_desc: 'Confirm your 2D concept to generate a 4-view multicolor inspection sheet.',
    cam_iso: '3/4 Iso', cam_front: 'Front', cam_side: 'Side', cam_top: 'Top', cam_wire: 'Wire', cam_spin: 'Spin',
    card_hw: '🖨️ Printer & AMS Color Budget', printer_label: 'Target Printer', ams_label: 'Your AMS Units (sets final color cap)',
    ams0_t: '0 AMS', ams0_s: 'Split/Mono', ams1_t: '1 AMS', ams1_s: '≤4 col', ams2_t: '2 AMS', ams2_s: '≤8 col', ams3_t: '3 AMS', ams3_s: '≤12 col', ams4_t: '4 AMS', ams4_s: '≤16 col',
    mode_label: 'Multicolor Strategy', mode_auto: 'Auto', mode_ams: 'AMS Single-Plate', mode_split: 'No-AMS Split (0 Poop)', mode_mono: 'Monochrome',
    joint_label: 'Glue-Free Mechanical Joint', joint_dt: 'Dovetail (Strongest)', joint_sp: 'Slotted Snap-Pin', joint_pt: 'Press-Fit Tenon', joint_none: 'None',
    card_dims: '📏 Dimensions (mm) & Filaments', add_color: '+ Add Color', dim_w: 'Width W', dim_d: 'Depth D', dim_h: 'Height H',
    swatch_label: 'Bambu PLA Swatches (click to append slot)', apply_spec: 'Apply & Refresh 2D Preview',
    card_dfm: '🛡️ DFM Printability & Color Decisions', m_mass: 'Filament', m_flush: 'AMS Flush', m_time: 'Est. Time',
    dfm_empty: 'Build the 3D model to view watertightness, overhangs, AMS color-merge logs, and split parts.',
    card_export: '📦 Export 3D Print Files', export_empty: 'Confirm design to generate exports…',
    slot_lbl: (s) => `Slot ${s}`, cap_no_ams: 'No AMS (Glue-Free Split / Mono)', cap_ams: (n) => `≤ ${n * 4} Colors`,
    welcome: 'Hi! Tell me what character or functional part you want to 3D print, its size in mm, and how many AMS units you have — or switch to the ready 3D WALL-E project above!',
    watertight_ok: '✅ Watertight Manifold', watertight_warn: '⚠️ Needs Slicer Repair',
    hud_mode: (d) => `${d.effective_multicolor_mode} · ${d.ams_slots_used} colors · joint: ${d.joint_type_used}`,
    status_3d_ready: '3D model ready — drag to orbit, scroll to zoom',
    status_loaded: (title) => `Loaded project: ${title}`,
    dl_3mf: '⬇ Bambu Multicolor .3MF', dl_3mf_tag: 'Recommended',
    dl_split: '⬇ No-AMS Glue-Free Plates .3MF', dl_split_tag: '0 Flush',
    dl_stl: '⬇ Universal Mesh .STL', dl_stl_tag: 'All Slicers',
    dl_step: '⬇ Parametric CAD .STEP', dl_step_tag: 'Editable',
    dl_py: '⬇ build123d Source .PY', dl_gcode: '⬇ Sliced .GCODE.3MF', dl_part: (k) => `⬇ Color Part ${k}.stl`,
    chips: [
      { label: '🤖 WALL-E Functional Ideas', prompt: 'I want a Pixar WALL-E robot with real desktop utility rather than just a figurine—what functional ideas do you suggest?' },
      { label: '🎧 No-AMS Dovetail Stand', prompt: 'Design a desktop headphone stand with rear cable slot, 210mm tall, matte black and orange, no AMS, glue-free dovetail split parts' },
      { label: '🦾 Iron Man MagSafe Dock', prompt: 'Design a stylized Marvel Iron Man chest bust with a MagSafe wireless charger pocket, 125mm tall, 1 AMS unit 4 colors' },
      { label: '📦 Gridfinity Organizer', prompt: 'Design a 3-compartment Gridfinity desk organizer, 150x100x85mm, 1 AMS unit 3 colors' },
    ],
  },
};

function t(key) {
  return (I18N[lang] && I18N[lang][key]) || I18N.zh[key] || key;
}

function renderQuickChips() {
  const box = $('quickChips');
  if (!box) return;
  box.innerHTML = '';
  for (const ch of t('chips')) {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'chip';
    b.textContent = ch.label;
    b.onclick = () => sendMessage(ch.prompt);
    box.appendChild(b);
  }
}

function renderMwQuickTags() {
  const box = $('mwQuickTags');
  if (!box) return;
  const tags = ['WALL-E', 'Gridfinity', 'Headphone Stand', 'Skadis', 'Iron Man', 'Totoro'];
  box.innerHTML = '';
  for (const tag of tags) {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'chip';
    b.textContent = tag;
    b.onclick = () => {
      $('mwQuery').value = tag;
      runMakerWorldSearch(tag);
    };
    box.appendChild(b);
  }
}

function applyLang(newLang) {
  lang = newLang === 'en' ? 'en' : 'zh';
  localStorage.setItem('pf_lang', lang);
  document.documentElement.lang = lang === 'en' ? 'en' : 'zh-CN';
  document.querySelectorAll('[data-i18n]').forEach((el) => {
    const k = el.dataset.i18n;
    const val = t(k);
    if (typeof val === 'string') el.textContent = val;
  });
  $('langToggleBtn').textContent = t('lang_btn');
  $('chatInput').placeholder = t('chat_ph');
  if ($('mwQuery')) $('mwQuery').placeholder = t('mw_ph');
  renderQuickChips();
  renderMwQuickTags();
  if (project) renderProject();
}

// Three.js scene state
let renderer, scene, camera, controls, currentModelGroup, bedGrid, lastModelSize = new THREE.Vector3(100, 100, 100);
const loader = new GLTFLoader();

const $ = (id) => document.getElementById(id);

function setStatus(msg) {
  $('statusBar').textContent = msg;
}

function setBusy(isBusy, title = '正在生成…', sub = 'RTX 5090 本地推理中，请稍候') {
  $('busyOverlay').classList.toggle('hidden', !isBusy);
  $('busyTitle').textContent = title;
  $('busySub').textContent = sub;
  $('sendBtn').disabled = isBusy;
  $('confirmBuildBtn').disabled = isBusy;
}

function updateStepper(stage) {
  const order = ['ideation', 'concept_review', 'model_ready', 'exported'];
  const idx = Math.max(0, order.indexOf(stage || 'ideation'));
  document.querySelectorAll('.step').forEach((el) => {
    const sIdx = order.indexOf(el.dataset.step);
    el.classList.remove('active', 'done');
    if (sIdx < idx) el.classList.add('done');
    else if (sIdx === idx) el.classList.add('active');
  });
}

function initThree() {
  const container = $('threeContainer');
  if (renderer) return;
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0f1522);

  camera = new THREE.PerspectiveCamera(40, container.clientWidth / Math.max(container.clientHeight, 1), 0.5, 4000);
  camera.position.set(190, 150, 210);

  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(container.clientWidth, container.clientHeight);
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.15;
  container.appendChild(renderer.domElement);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.autoRotate = autoSpin;
  controls.autoRotateSpeed = 1.4;

  scene.add(new THREE.HemisphereLight(0xf6f9ff, 0x263147, 1.45));
  const key = new THREE.DirectionalLight(0xffffff, 1.6);
  key.position.set(180, 280, 190);
  scene.add(key);
  const rim = new THREE.DirectionalLight(0x8ec0ff, 0.85);
  rim.position.set(-190, 130, -170);
  scene.add(rim);

  drawBuildPlate(256);

  window.addEventListener('resize', () => {
    if (!renderer) return;
    const w = container.clientWidth;
    const h = Math.max(container.clientHeight, 1);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
  });

  const animate = () => {
    requestAnimationFrame(animate);
    controls.autoRotate = autoSpin;
    controls.update();
    renderer.render(scene, camera);
  };
  animate();
}

function drawBuildPlate(sizeMM = 256) {
  if (!scene) return;
  if (bedGrid) scene.remove(bedGrid);
  bedGrid = new THREE.GridHelper(sizeMM, Math.round(sizeMM / 16), 0x4b6a96, 0x222f46);
  scene.add(bedGrid);
}

function setCameraPreset(preset) {
  if (!camera || !controls) return;
  const maxDim = Math.max(lastModelSize.x, lastModelSize.y, lastModelSize.z, 70);
  const cy = lastModelSize.y * 0.45;
  controls.target.set(0, cy, 0);
  if (preset === 'front') camera.position.set(0, cy * 1.05, maxDim * 2.0);
  else if (preset === 'side') camera.position.set(maxDim * 2.0, cy * 1.05, 0);
  else if (preset === 'top') camera.position.set(0, maxDim * 2.2, 0.01);
  else camera.position.set(maxDim * 1.4, maxDim * 1.1, maxDim * 1.45);
  controls.update();
}

function applyWireframe() {
  if (!currentModelGroup) return;
  currentModelGroup.traverse((obj) => {
    if (obj.isMesh && obj.material) {
      obj.material.wireframe = wireframeOn;
    }
  });
}

function loadGlbPreview(url) {
  if (!url) return;
  initThree();
  setStatus(`加载 3D 交互模型 (${url})…`);
  loader.load(
    `${url}?t=${Date.now()}`,
    (gltf) => {
      if (currentModelGroup) scene.remove(currentModelGroup);
      const wrapper = new THREE.Group();
      gltf.scene.rotation.x = -Math.PI / 2;
      wrapper.add(gltf.scene);

      const box = new THREE.Box3().setFromObject(wrapper);
      const center = box.getCenter(new THREE.Vector3());
      const size = box.getSize(new THREE.Vector3());
      wrapper.position.set(-center.x, -box.min.y, -center.z);
      scene.add(wrapper);
      currentModelGroup = wrapper;
      lastModelSize = size.clone();
      applyWireframe();
      setCameraPreset('iso');

      $('hudDims').textContent = `${size.x.toFixed(1)} × ${size.z.toFixed(1)} × ${size.y.toFixed(1)} mm`;
      const d = project?.dfm;
      $('hudWatertight').textContent = d ? (d.watertight ? t('watertight_ok') : t('watertight_warn')) : '3D';
      $('hudMode').textContent = d ? t('hud_mode')(d) : 'Three.js';
      setStatus(t('status_3d_ready'));
    },
    undefined,
    (err) => setStatus(`3D 预览加载失败: ${err.message || err}`)
  );
}

function switchView(view) {
  activeView = view;
  document.querySelectorAll('.vpill').forEach((el) => el.classList.toggle('active', el.dataset.view === view));
  $('view2d').classList.toggle('active', view === '2d');
  $('view3d').classList.toggle('active', view === '3d' || view === 'split');
  $('viewTurntable').classList.toggle('active', view === 'turntable');

  if (view === '3d' && project?.exports?.urls?.preview_glb) {
    loadGlbPreview(project.exports.urls.preview_glb);
  } else if (view === 'split') {
    const url = project?.exports?.urls?.split_preview_glb || project?.exports?.urls?.preview_glb;
    if (url) loadGlbPreview(url);
  }
}

function updateAmsPills(amsUnits) {
  document.querySelectorAll('.ams-pill').forEach((b) => {
    b.classList.toggle('active', Number(b.dataset.ams) === Number(amsUnits));
  });
  const badge = $('colorCapBadge');
  badge.textContent = amsUnits === 0 ? t('cap_no_ams') : t('cap_ams')(amsUnits);
}

function renderProject() {
  if (!project) return;
  const spec = project.spec;
  $('projectIdTag').textContent = project.project_id;
  updateStepper(project.stage);

  // Chat log (also embed latest 2D concept preview card inline so users NEVER miss the 2D render!)
  const log = $('chatLog');
  log.innerHTML = '';
  if (!project.history.length) {
    log.innerHTML = `<div class="msg assistant">${t('welcome')}</div>`;
  } else {
    for (const turn of project.history) {
      const div = document.createElement('div');
      div.className = `msg ${turn.role}`;
      div.textContent = turn.content;
      log.appendChild(div);
    }
  }
  const latestConcept = project.concepts?.[project.concepts.length - 1];
  if (latestConcept?.image_url) {
    const card = document.createElement('div');
    card.className = 'chat-concept-card';
    const imgSrc = `${latestConcept.image_url}?t=${Date.now()}`;
    card.innerHTML = `
      <img src="${imgSrc}" alt="2D Concept Render" title="${t('inline_zoom')}" />
      <div class="chat-concept-foot">
        <span>🎨 ${t('inline_2d_label')} #${project.concepts.length} (${spec.dimensions.width_mm}×${spec.dimensions.depth_mm}×${spec.dimensions.height_mm} mm)</span>
        <div class="chat-concept-actions">
          <button type="button" class="mini-btn zoom-2d-btn">${t('inline_zoom')}</button>
          <button type="button" class="mini-btn accent build-3d-btn">${t('inline_build3d')}</button>
        </div>
      </div>
    `;
    const imgEl = card.querySelector('img');
    imgEl.onload = () => { log.scrollTop = log.scrollHeight; };
    imgEl.onclick = () => {
      switchView('2d');
      if (window.innerWidth <= 768) setMobileTab('canvas');
    };
    card.querySelector('.zoom-2d-btn').onclick = () => {
      switchView('2d');
      if (window.innerWidth <= 768) setMobileTab('canvas');
    };
    card.querySelector('.build-3d-btn').onclick = () => {
      if (window.innerWidth <= 768) setMobileTab('canvas');
      $('confirmBuildBtn').click();
    };
    log.appendChild(card);
  }
  log.scrollTop = log.scrollHeight;
  $('refCount').textContent = t('refs_count')(project.references.length);

  // Attached refs list
  const att = $('attachedRefs');
  att.innerHTML = project.references.length
    ? project.references.map((r) => `<div class="mw-item"><span>${r.title}</span><span class="muted">${r.source}</span></div>`).join('')
    : `<div class="muted">${t('no_refs')}</div>`;

  // Right-pane controls
  $('printerSelect').value = spec.printer_id;
  updateAmsPills(spec.ams_units);
  $('multicolorModeSelect').value = spec.multicolor_mode;
  $('jointTypeSelect').value = spec.joint_type;
  $('dimW').value = spec.dimensions.width_mm;
  $('dimD').value = spec.dimensions.depth_mm;
  $('dimH').value = spec.dimensions.height_mm;

  const pInfo = meta?.printers?.[spec.printer_id];
  if (pInfo && scene) drawBuildPlate(Math.max(pInfo.bed_x_mm, pInfo.bed_y_mm));

  // Filaments editor
  const flBox = $('filamentList');
  flBox.innerHTML = '';
  spec.filaments.forEach((fl, idx) => {
    const row = document.createElement('div');
    row.className = 'fl-row';
    row.innerHTML = `
      <input type="color" value="${fl.hex_color}" data-idx="${idx}" class="fl-color" />
      <input type="text" value="${fl.name}" data-idx="${idx}" class="fl-name" />
      <span class="muted">${t('slot_lbl')(fl.slot)}</span>
      <button class="btn-link fl-del" data-idx="${idx}" title="Remove">×</button>
    `;
    flBox.appendChild(row);
  });

  // 2D Concept Gate
  const latest = project.concepts?.[project.concepts.length - 1];
  if (latest?.image_url) {
    $('conceptPlaceholder').classList.add('hidden');
    $('conceptImg').classList.remove('hidden');
    $('conceptImg').src = `${latest.image_url}?t=${Date.now()}`;
    $('conceptCaption').innerHTML = `<span>${latest.backend}</span><span>${spec.dimensions.width_mm}×${spec.dimensions.depth_mm}×${spec.dimensions.height_mm} mm · ${spec.filaments.length} colors</span>`;
  } else {
    $('conceptPlaceholder').classList.remove('hidden');
    $('conceptImg').classList.add('hidden');
    $('conceptCaption').textContent = '';
  }

  // 4-angle turntable
  const ttUrl = project.exports?.urls?.turntable_preview;
  if (ttUrl) {
    $('turntablePlaceholder').classList.add('hidden');
    $('turntableImg').classList.remove('hidden');
    $('turntableImg').src = `${ttUrl}?t=${Date.now()}`;
    $('turntableCaption').textContent = `${spec.title} — 3/4 · Front · Side · Rear`;
  } else {
    $('turntablePlaceholder').classList.remove('hidden');
    $('turntableImg').classList.add('hidden');
    $('turntableCaption').textContent = '';
  }

  // DFM & Color Decisions
  const dfmBox = $('dfmSummary');
  if (project.dfm) {
    const d = project.dfm;
    const exp = project.exports || {};
    $('dfmMetrics').classList.remove('hidden');
    $('mMass').textContent = `${exp.estimated_filament_g ?? d.estimated_mass_g} g`;
    $('mFlush').textContent = `${exp.estimated_flush_g ?? 0} g`;
    $('mTime').textContent = exp.estimated_print_minutes ? `${Math.round(exp.estimated_print_minutes)} min` : '-';
    $('hudWatertight').textContent = d.watertight ? t('watertight_ok') : t('watertight_warn');
    $('hudMode').textContent = t('hud_mode')(d);
    if (activeView === '3d' || activeView === 'split') setStatus(t('status_3d_ready'));

    let html = '';
    for (const c of d.checks || []) {
      html += `<div class="check-pill ${c.status}"><strong>${c.name}:</strong> ${c.detail}</div>`;
    }
    if (d.color_decisions?.length) {
      html += `<div class="micro-label mt-4">AMS Color Budget (${d.ams_units_configured} AMS):</div>`;
      for (const cd of d.color_decisions) {
        html += `<div class="muted">• ${cd.source_hex} → ${t('slot_lbl')(cd.mapped_slot)} ${cd.mapped_name} (${cd.action}${cd.delta_e ? `, ΔE=${cd.delta_e}` : ''})</div>`;
      }
    }
    if (d.split_parts?.length > 1) {
      html += `<div class="micro-label mt-4">Split Parts (${d.split_parts.length}):</div>`;
      for (const sp of d.split_parts) {
        html += `<div class="muted">• Plate ${sp.plate_index} [${t('slot_lbl')(sp.slot)}] ${sp.name} (${sp.bbox_mm.join('×')} mm · ${sp.joint_type})</div>`;
      }
    }
    dfmBox.innerHTML = html;
  } else {
    $('dfmMetrics').classList.add('hidden');
    dfmBox.textContent = t('dfm_empty');
  }

  // Exports
  const expBox = $('exportLinks');
  const urls = project.exports?.urls || {};
  const links = [];
  if (urls.threemf) links.push(`<a class="dl-btn primary" href="${urls.threemf}" download>${t('dl_3mf')} <span class="dl-badge">${t('dl_3mf_tag')}</span></a>`);
  if (urls.split_plates_3mf) links.push(`<a class="dl-btn primary" href="${urls.split_plates_3mf}" download>${t('dl_split')} <span class="dl-badge">${t('dl_split_tag')}</span></a>`);
  if (urls.stl) links.push(`<a class="dl-btn" href="${urls.stl}" download>${t('dl_stl')} <span class="dl-badge">${t('dl_stl_tag')}</span></a>`);
  if (urls.step) links.push(`<a class="dl-btn" href="${urls.step}" download>${t('dl_step')} <span class="dl-badge">${t('dl_step_tag')}</span></a>`);
  if (urls.cad_script) links.push(`<a class="dl-btn" href="${urls.cad_script}" download>${t('dl_py')} <span class="dl-badge">Python</span></a>`);
  if (urls.gcode_3mf) links.push(`<a class="dl-btn primary" href="${urls.gcode_3mf}" download>${t('dl_gcode')} <span class="dl-badge">CLI</span></a>`);
  for (const [k, u] of Object.entries(urls.per_color_stls || {})) {
    if (u) links.push(`<a class="dl-btn" href="${u}" download>${t('dl_part')(k)}</a>`);
  }
  expBox.innerHTML = links.length ? links.join('') : `<span class="muted">${t('export_empty')}</span>`;
}

function collectSpecFromUI() {
  const fls = [];
  document.querySelectorAll('#filamentList .fl-row').forEach((row, idx) => {
    fls.push({
      slot: idx + 1,
      hex_color: row.querySelector('.fl-color').value.toUpperCase(),
      name: row.querySelector('.fl-name').value || `Color ${idx + 1}`,
      material: 'PLA',
      role: idx === 0 ? 'main body' : `accent ${idx}`,
    });
  });
  const activeAmsBtn = document.querySelector('.ams-pill.active');
  return {
    printer_id: $('printerSelect').value,
    ams_units: activeAmsBtn ? Number(activeAmsBtn.dataset.ams) : 1,
    multicolor_mode: $('multicolorModeSelect').value,
    joint_type: $('jointTypeSelect').value,
    dimensions: {
      width_mm: Number($('dimW').value),
      depth_mm: Number($('dimD').value),
      height_mm: Number($('dimH').value),
    },
    filaments: fls.length ? fls : project.spec.filaments,
  };
}

async function refreshProjectList(selectedId = null) {
  const r = await fetch('/api/projects');
  const list = await r.json();
  const sel = $('projectSwitcher');
  sel.innerHTML = '';
  for (const item of list) {
    const opt = document.createElement('option');
    opt.value = item.project_id;
    opt.textContent = `${item.title} (${item.project_id})`;
    sel.appendChild(opt);
  }
  if (selectedId) sel.value = selectedId;
  return list;
}

async function loadProject(projectId) {
  const r = await fetch(`/api/projects/${projectId}`);
  project = await r.json();
  renderProject();
  // Show 2D concept if stage is concept_review (so user sees latest 2D render!), else 3D if built
  if (project.stage === 'concept_review' && project.concepts?.length) {
    switchView('2d');
  } else if (project.exports?.urls?.preview_glb) {
    switchView('3d');
  } else {
    switchView('2d');
  }
  setStatus(t('status_loaded')(project.spec.title));
}

async function createNewProject() {
  const r = await fetch('/api/projects', { method: 'POST' });
  project = await r.json();
  await refreshProjectList(project.project_id);
  renderProject();
  switchView('2d');
  setStatus(`已新建空白项目 ${project.project_id}`);
}

let isSendingChat = false;

async function sendMessage(text) {
  const clean = (text || '').trim();
  if (!clean) return;
  if (isSendingChat) return;

  // Ensure a project exists even if user types before bootstrap finished
  if (!project) {
    await createNewProject();
  }

  isSendingChat = true;
  $('sendBtn').disabled = true;

  // 1. Optimistic immediate render so user NEVER feels the textbox froze or swallowed their text!
  const log = $('chatLog');
  const userDiv = document.createElement('div');
  userDiv.className = 'msg user';
  userDiv.textContent = clean;
  log.appendChild(userDiv);

  const typingDiv = document.createElement('div');
  typingDiv.className = 'msg assistant typing';
  typingDiv.textContent = t('thinking_msg');
  log.appendChild(typingDiv);
  log.scrollTop = log.scrollHeight;
  setStatus(t('thinking_msg'));

  try {
    // Step A: Fast text & spec response (~1-2s, never blocked by 20s SDXL generation!)
    const r = await fetch(`/api/projects/${project.project_id}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: clean, auto_render_2d: false }),
    });
    if (!r.ok) throw new Error(`HTTP ${r.status}: ${await r.text()}`);
    const data = await r.json();
    project = data.project;
    await refreshProjectList(project.project_id);
    renderProject();

    // Unlock chat input immediately so user can read/keep chatting!
    isSendingChat = false;
    $('sendBtn').disabled = false;

    // Step B: Async 2D concept rendering if enabled (only shows spinner on center canvas, never locks chat!)
    const want2d = $('auto2dCheck') ? $('auto2dCheck').checked : true;
    if (want2d) {
      switchView('2d');
      setStatus(t('rendering_2d_msg'));
      $('busyOverlay').classList.remove('hidden');
      $('busyTitle').textContent = lang === 'en' ? 'Rendering 2D Concept…' : '正在生成 2D 效果图…';
      $('busySub').textContent = lang === 'en' ? 'Local GPU SDXL / Studio renderer (chat remains unlocked)' : '本地 GPU SDXL / 工作室引擎渲染中（可继续对话）';
      try {
        const r2 = await fetch(`/api/projects/${project.project_id}/render-2d`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({}),
        });
        if (r2.ok) {
          project = await r2.json();
          renderProject();
          switchView('2d');
          setStatus(lang === 'en' ? '2D concept updated — confirm to build 3D print files.' : '2D 效果图已更新 — 满意后点右上角生成 3D 打印文件');
        }
      } finally {
        $('busyOverlay').classList.add('hidden');
      }
    }
  } catch (err) {
    typingDiv.className = 'msg error';
    typingDiv.textContent = `Error: ${err.message || err}`;
    $('chatInput').value = clean;
    setStatus(`Error: ${err.message || err}`);
    isSendingChat = false;
    $('sendBtn').disabled = false;
  }
}

async function bootstrap() {
  const r = await fetch('/api/meta');
  meta = await r.json();
  const d = meta.doctor;
  const gpuShort = d.gpu_name ? d.gpu_name.replace('NVIDIA GeForce ', '').replace(' Laptop GPU', '') : d.accelerator;
  $('hwBadge').textContent = `⚡ ${gpuShort} · ${d.recommended_llm || 'Fallback'}`;

  // Printers
  const ps = $('printerSelect');
  ps.innerHTML = '';
  for (const p of Object.values(meta.printers)) {
    const opt = document.createElement('option');
    opt.value = p.id;
    opt.textContent = `${p.name} (${p.bed_x_mm}×${p.bed_y_mm}×${p.bed_z_mm})`;
    ps.appendChild(opt);
  }

  // Bambu official PLA swatches
  const sw = $('bambuSwatches');
  sw.innerHTML = '';
  for (const [name, hx] of Object.entries(meta.swatches || {})) {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'swatch-dot';
    b.style.background = hx;
    b.title = `${name} (${hx})`;
    b.onclick = () => {
      if (!project) return;
      const fls = project.spec.filaments;
      if (fls.length < 16) {
        fls.push({ slot: fls.length + 1, name, hex_color: hx, material: 'PLA', role: `accent ${fls.length}` });
        renderProject();
      }
    };
    sw.appendChild(b);
  }

  // Bilingual UI init
  applyLang(lang);

  // Open latest showcased project (e.g. live_walle_v2) immediately if available so user sees real 3D right away!
  const existing = await refreshProjectList();
  const preferred = existing.find((p) => p.project_id === 'live_walle_v2')
    || existing.find((p) => p.project_id === 'demo_no_ams_stand')
    || existing[0];
  if (preferred) {
    $('projectSwitcher').value = preferred.project_id;
    await loadProject(preferred.project_id);
  } else {
    await createNewProject();
  }
}

// Event bindings
$('newProjectBtn').onclick = createNewProject;
$('projectSwitcher').onchange = (e) => loadProject(e.target.value);
$('langToggleBtn').onclick = () => applyLang(lang === 'zh' ? 'en' : 'zh');
$('toggleInspectorBtn').onclick = () => {
  const ws = document.querySelector('.workspace');
  if (window.innerWidth <= 1020) {
    ws.classList.toggle('inspector-open-overlay');
  } else {
    ws.classList.toggle('inspector-collapsed');
  }
};

document.querySelectorAll('.left-tab').forEach((btn) => {
  btn.onclick = () => {
    document.querySelectorAll('.left-tab').forEach((b) => b.classList.toggle('active', b === btn));
    document.querySelectorAll('.left-panel').forEach((p) => p.classList.toggle('active', p.id === `panel-${btn.dataset.panel}`));
  };
});

document.querySelectorAll('.vpill').forEach((el) => {
  el.onclick = () => switchView(el.dataset.view);
});

document.querySelectorAll('.cam-btn[data-cam]').forEach((btn) => {
  btn.onclick = () => setCameraPreset(btn.dataset.cam);
});
$('wireframeBtn').onclick = () => {
  wireframeOn = !wireframeOn;
  $('wireframeBtn').classList.toggle('active', wireframeOn);
  applyWireframe();
};
$('spinBtn').onclick = () => {
  autoSpin = !autoSpin;
  $('spinBtn').classList.toggle('active', autoSpin);
};

document.querySelectorAll('.ams-pill').forEach((btn) => {
  btn.onclick = () => {
    const n = Number(btn.dataset.ams);
    updateAmsPills(n);
    if (n === 0 && $('multicolorModeSelect').value === 'ams') {
      $('multicolorModeSelect').value = 'split_no_ams';
    }
  };
});

// Quick chips are bound dynamically inside renderQuickChips()

function submitChatFromInput() {
  const el = $('chatInput');
  const text = el.value.trim();
  if (!text || isSendingChat) return;
  el.value = '';
  sendMessage(text);
}

$('chatForm').addEventListener('submit', (e) => {
  e.preventDefault();
  submitChatFromInput();
});

$('chatInput').addEventListener('keydown', (e) => {
  // Never hijack Enter during Chinese/Japanese/Korean IME composition!
  if (e.key === 'Enter' && !e.shiftKey && !e.isComposing && e.keyCode !== 229) {
    e.preventDefault();
    submitChatFromInput();
  }
});

// Allow Ctrl+V pasting a MakerWorld screenshot straight into the chat input!
$('chatInput').addEventListener('paste', async (e) => {
  const items = e.clipboardData?.items || [];
  for (const item of items) {
    if (item.type && item.type.startsWith('image/')) {
      e.preventDefault();
      const blob = item.getAsFile();
      if (blob && project) {
        await uploadReferenceBlob(blob, `pasted_ref_${Date.now()}.png`);
      }
      return;
    }
  }
});

$('addColorBtn').onclick = () => {
  if (!project || project.spec.filaments.length >= 16) return;
  const defaults = ['#1E4C8A', '#F2EFE9', '#4D9A4E', '#F4D03F', '#C9302C', '#A28BC7'];
  const nextSlot = project.spec.filaments.length + 1;
  project.spec.filaments.push({
    slot: nextSlot,
    name: `AMS Color ${nextSlot}`,
    hex_color: defaults[(nextSlot - 1) % defaults.length],
    material: 'PLA',
    role: `accent ${nextSlot}`,
  });
  renderProject();
};

$('filamentList').onclick = (e) => {
  if (!e.target.classList.contains('fl-del')) return;
  const idx = Number(e.target.dataset.idx);
  if (project.spec.filaments.length > 1) {
    project.spec.filaments.splice(idx, 1);
    project.spec.filaments.forEach((f, i) => (f.slot = i + 1));
    renderProject();
  }
};

$('applySpecBtn').onclick = async () => {
  setBusy(true, '正在更新规格与 2D 概念图…', '应用最新尺寸、AMS 数量与耗材色盘');
  try {
    const patch = collectSpecFromUI();
    const r = await fetch(`/api/projects/${project.project_id}/spec`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ patch, re_render_2d: true }),
    });
    project = await r.json();
    renderProject();
    switchView('2d');
    setStatus('规格与 2D 概念图已刷新');
  } finally {
    setBusy(false);
  }
};

$('confirmBuildBtn').onclick = async () => {
  if (!project) return;
  setBusy(true, '正在生成 3D 模型、水密修复与 .3MF 打包…', 'CAD / Hunyuan3D-2 + 布尔功能雕刻 + AMS 颜色映射');
  try {
    const patch = collectSpecFromUI();
    await fetch(`/api/projects/${project.project_id}/spec`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ patch, re_render_2d: false }),
    });
    const r = await fetch(`/api/projects/${project.project_id}/confirm-build`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ run_slicer: true }),
    });
    project = await r.json();
    await refreshProjectList(project.project_id);
    renderProject();
    switchView(project.exports?.urls?.split_preview_glb ? 'split' : '3d');
    setStatus('生成完成！可直接在画布旋转检查或在右侧下载 .3MF / .STL / .STEP');
  } finally {
    setBusy(false);
  }
};

async function runMakerWorldSearch(q) {
  const query = (q || '').trim();
  if (!query) return;
  $('mwResults').innerHTML = `<div class="muted">${lang === 'en' ? 'Searching MakerWorld…' : '正在搜索 MakerWorld 模型…'}</div>`;
  try {
    const r = await fetch(`/api/makerworld/search?q=${encodeURIComponent(query)}&limit=8`);
    const items = await r.json();
    $('mwResults').innerHTML = '';
    for (const it of items) {
      const card = document.createElement('div');
      card.className = 'mw-card';
      const thumbHtml = it.thumbnail_url ? `<img class="mw-thumb" src="${it.thumbnail_url}" alt="" onerror="this.style.display='none'" />` : '';
      card.innerHTML = `
        ${thumbHtml}
        <div class="mw-body">
          <a class="mw-title" href="${it.url}" target="_blank" rel="noopener" title="${it.title}">${it.title}</a>
          <div class="mw-snip">${it.snippet || ''}</div>
          <div class="mw-actions">
            <a class="btn-link" href="${it.url}" target="_blank" rel="noopener">${t('open_mw_btn')} ↗</a>
            <button type="button" class="btn-link mw-use-btn">${t('use_ref_btn')}</button>
          </div>
        </div>
      `;
      card.querySelector('.mw-use-btn').onclick = async () => {
        if (!project) await createNewProject();
        const resp = await fetch(`/api/projects/${project.project_id}/references/url`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ url: it.url }),
        });
        const data = await resp.json();
        project = data.project;
        renderProject();
        document.querySelector('.left-tab[data-panel="chat"]').click();
        setStatus(lang === 'en' ? `Attached MakerWorld reference: ${it.title}` : `已引用 MakerWorld 参考：${it.title}`);
      };
      $('mwResults').appendChild(card);
    }
  } catch (err) {
    $('mwResults').innerHTML = `<div class="muted">Search error: ${err.message}</div>`;
  }
}

$('mwSearchBtn').onclick = () => runMakerWorldSearch($('mwQuery').value);
$('mwQuery').addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.isComposing && e.keyCode !== 229) {
    e.preventDefault();
    runMakerWorldSearch($('mwQuery').value);
  }
});

$('mwUrlBtn').onclick = async () => {
  const url = $('mwUrlInput').value.trim();
  if (!url) return;
  if (!project) await createNewProject();
  const r = await fetch(`/api/projects/${project.project_id}/references/url`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ url }),
  });
  const data = await r.json();
  project = data.project;
  $('mwUrlInput').value = '';
  renderProject();
  setStatus(lang === 'en' ? `Imported URL reference: ${data.reference.title}` : `已导入参考链接：${data.reference.title}`);
};

async function uploadReferenceBlob(fileOrBlob, filename) {
  if (!project) await createNewProject();
  setBusy(true, lang === 'en' ? `Analyzing reference ${filename}…` : `正在解析参考文件 ${filename}…`, 'VLM + K-Means AMS Palette');
  try {
    const fd = new FormData();
    fd.append('file', fileOrBlob, filename);
    const r = await fetch(`/api/projects/${project.project_id}/references/upload`, { method: 'POST', body: fd });
    const data = await r.json();
    project = data.project;
    renderProject();
    setStatus(lang === 'en' ? `Extracted reference palette from ${filename}` : `已从 ${filename} 提取参考色盘与特征`);
  } finally {
    setBusy(false);
  }
}

$('refUpload').onchange = async (e) => {
  const file = e.target.files?.[0];
  if (!file) return;
  await uploadReferenceBlob(file, file.name);
  e.target.value = '';
};

function setMobileTab(tab) {
  document.body.classList.remove('mtab-chat', 'mtab-canvas', 'mtab-makerworld', 'mtab-specs');
  document.body.classList.add(`mtab-${tab}`);
  document.querySelectorAll('.mnav-btn').forEach((b) => {
    b.classList.toggle('active', b.dataset.mtab === tab);
  });
  if (tab === 'chat' || tab === 'makerworld') {
    document.querySelectorAll('.left-tab').forEach((b) => b.classList.toggle('active', b.dataset.panel === tab));
    document.querySelectorAll('.left-panel').forEach((p) => p.classList.toggle('active', p.id === `panel-${tab}`));
  }
  if (tab === 'canvas') {
    setTimeout(() => window.dispatchEvent(new Event('resize')), 50);
  }
}

async function openMobileModal() {
  const modal = $('mobileModal');
  modal.classList.remove('hidden');
  try {
    const r = await fetch('/api/lan-info');
    const info = await r.json();
    const pill = $('lanStatusPill');
    pill.textContent = info.lan_enabled ? t('lan_ok') : t('lan_local_only');
    pill.classList.toggle('warn', !info.lan_enabled);
    $('lanQrBox').innerHTML = info.qr_svg || '<div class="muted">QR unavailable</div>';
    $('lanUrlInput').value = info.lan_url;
    $('lanHintBox').textContent = info.lan_enabled
      ? t('lan_hint_ok')(info.primary_ip)
      : t('lan_hint_local')(info.cli_hint);
  } catch (err) {
    $('lanStatusPill').textContent = `Error: ${err.message || err}`;
  }
}

$('mobilePairBtn').onclick = openMobileModal;
$('closeMobileModalBtn').onclick = () => $('mobileModal').classList.add('hidden');
$('mobileModal').addEventListener('click', (e) => {
  if (e.target === $('mobileModal')) $('mobileModal').classList.add('hidden');
});
$('copyLanUrlBtn').onclick = async () => {
  const val = $('lanUrlInput').value;
  try {
    await navigator.clipboard.writeText(val);
    $('copyLanUrlBtn').textContent = t('copied_btn');
    setTimeout(() => ($('copyLanUrlBtn').textContent = t('copy_btn')), 1500);
  } catch {
    $('lanUrlInput').select();
    document.execCommand('copy');
  }
};

document.querySelectorAll('.mnav-btn').forEach((btn) => {
  btn.onclick = () => setMobileTab(btn.dataset.mtab);
});

setMobileTab('chat');
if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('/sw.js').catch(() => {});
}

bootstrap();
