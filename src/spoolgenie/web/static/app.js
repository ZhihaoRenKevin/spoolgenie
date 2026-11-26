import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

let project = null;
let meta = null;
let lang = localStorage.getItem('sg_lang') || ((navigator.language || '').toLowerCase().startsWith('zh') ? 'zh' : 'en');
let theme = document.documentElement.getAttribute('data-theme') || 'dark';
let isSendingChat = false;
let isRendering2D = false;
let isBuilding3D = false;
let runSlicerPref = true;
let active3DTab = '3d';
let autoSpin = true;
let wireframeOn = false;
let lampRubCount = 0;

let threeCtx = null;
const gltfLoader = new GLTFLoader();

const $ = (id) => document.getElementById(id);

const I18N = {
  zh: {
    brand_tag: '线盘神灯',
    new_btn: '✨ 新许愿',
    mw_btn: '🔎 MakerWorld',
    mobile_btn: '📱 手机',
    theme_to_dark: '🌙 星夜深色',
    theme_to_light: '☀️ 晨光浅色',
    lang_btn: 'EN',
    chat_ph: '对着神灯说出你的 3D 打印愿望…（Enter 许愿，支持 Ctrl+V 粘贴参考图）',
    attach_ref: '📎 参考图/STL',
    mw_chip: '🔎 MakerWorld 灵感',
    auto_2d: '自动显化 2D 幻象',
    send_btn: '✨ 许下愿望',
    lamp_whisper_idle: '✨ 轻点擦拭线盘神灯 · 唤醒 3D 具现魔法',
    lamp_whispers: [
      '🧞‍♂️ 线盘神灯已苏醒！说出你想具现的物件、角色或实用巧思吧～',
      '✨ 嗡——多色耗材线盘开始旋转，你的下一个愿望会落在拓竹热床上！',
      '🔮 告诉我尺寸和想法，剩下的 2D 幻象、免胶水卡扣与 .3MF 交给我！',
    ],
    hero_title: '擦亮线盘神灯，说出你的 3D 愿望',
    hero_sub: '只需在这里自然聊天许愿。显化 2D 幻象后，打印机型号、AMS 台数、耗材色盘、免胶水卡扣与切片选项会自动出现在对话流里供你点选。',
    starters: [
      { title: '📜 许愿 · 瓦力功能桌面摆件', desc: '保留瓦力萌眼，背部精准雕刻笔筒或手机槽', prompt: '我想生成一个机器人总动员里面的瓦力，但是又不想完全弄成手办想有一些功能，你有啥建议' },
      { title: '📜 许愿 · 无 AMS 免胶水耳机架', desc: '哑光黑 + 橘红双色，燕尾槽免胶水分盘 0 废料', prompt: '做一个桌面耳机支架，带底部理线槽，高210mm，哑光黑+橘红，没有AMS，帮我用燕尾槽免胶水拆件打印' },
      { title: '📜 许愿 · 钢铁侠 MagSafe 充座', desc: 'Q 版战甲胸口内嵌 MagSafe 磁吸槽，1 台 AMS 四色', prompt: '做一个漫威钢铁侠Q版战甲胸口MagSafe无线充底座，高125mm，1台AMS四色打印' },
      { title: '📜 许愿 · 头顶橘子卡皮巴拉', desc: '呆萌水豚摆件，内置 171 项静默主题与 19 项工程先验', prompt: '做一个头顶橘子的卡皮巴拉水豚桌面摆件，高115mm，1台AMS四色打印' },
    ],
    card_2d_title: '🔮 愿望幻象 (2D) & 具现仪式配置',
    zoom_btn: '🔍 放大端详',
    sec_printer: '🖨️ 1. 选择你的打印法器（机型）',
    sec_ams: '🌈 2. 选择 AMS 料盘数与多色方案',
    sec_joint: '🧩 3. 免胶水机械连接 / 分盘法术',
    sec_dims_color: '📏 4. 尺寸 (mm) 与耗材颜色',
    sec_slice: '🔪 5. 输出与自动切片选项',
    ams_opts: [
      { n: 0, t: '无 AMS (0 台)', s: '分色拆件 · 0 废料' },
      { n: 1, t: '1 台 AMS', s: '最多 4 色' },
      { n: 2, t: '2 台 AMS', s: '最多 8 色' },
      { n: 3, t: '3 台 AMS', s: '最多 12 色' },
      { n: 4, t: '4 台 AMS', s: '最多 16 色' },
    ],
    mode_opts: [
      { v: 'auto', t: '自动策略' },
      { v: 'ams', t: 'AMS 同盘换色' },
      { v: 'split_no_ams', t: '无 AMS 分盘 (0废料)' },
      { v: 'monochrome', t: '合并单色' },
    ],
    joint_opts: [
      { v: 'dovetail', t: '燕尾槽', s: '最稳抗拔' },
      { v: 'snap_pin', t: '弹性卡扣销', s: '按压即扣' },
      { v: 'press_tenon', t: '紧配榫卯', s: '隐藏平齐' },
      { v: 'none', t: '无接头', s: '单件/同盘' },
    ],
    slice_opts: [
      { v: true, t: '铸造 .3MF + 自动切片', s: '检测本地 Bambu/Orca CLI' },
      { v: false, t: '仅铸造 .3MF / .STL / .STEP', s: '拖入切片软件精调' },
    ],
    add_color: '+ 加一色',
    swatch_hint: '点击拓竹 PLA 色珠快速加入：',
    refresh_2d_btn: '🔄 应用修改 · 重绘 2D 幻象',
    build_3d_btn: '✨ 吟唱具现 · 铸造 3D 与 .3MF',
    rendering_2d_inline: '🧞‍♂️ 神灯正在凝结 2D 愿望幻象，你可以继续补充想法…',
    building_3d_inline: '✨ 魔法阵运转中 · 正在铸造水密 3D 实体、免胶水卡扣与拓竹 .3MF…',
    thinking_msg: '🧞‍♂️ 神灯正在聆听你的愿望并构思方案…',
    card_3d_title: '🏆 愿望已具现 · 3D 实体与打印文件',
    tab_3d: '🧊 3D 实体',
    tab_split: '🧩 无 AMS 分盘',
    tab_tt: '📐 四视角验收图',
    cam_iso: '3/4', cam_front: '正面', cam_side: '侧面', cam_top: '俯视', cam_wire: '线框', cam_spin: '旋转',
    m_watertight: '几何水密', m_mass: '耗材质量', m_flush: '冲刷废料', m_time: '预估耗时',
    wt_ok: '✅ 水密可打', wt_warn: '⚠️ 需修补',
    dfm_toggle: '展开 DFM 体检、AMS 颜色取舍与分块清单',
    dl_3mf: '⬇ 拓竹多色 .3MF', dl_3mf_tag: '直接切片',
    dl_split: '⬇ 无AMS免胶水分盘 .3MF', dl_split_tag: '0 废料',
    dl_stl: '⬇ 通用网格 .STL', dl_stl_tag: '全切片器',
    dl_step: '⬇ 参数化 CAD .STEP', dl_step_tag: '可编辑',
    dl_py: '⬇ build123d 源码 .PY', dl_gcode: '⬇ 已切片 .GCODE.3MF',
    dl_part: (k) => `⬇ 分色零件 ${k}.stl`,
    mw_drawer_title: '🔎 MakerWorld 灵感宝库搜索',
    mw_note: '搜索公开 MakerWorld 模型并一键带入许愿对话，或粘贴模型链接提取结构与色盘。',
    mw_ph: '搜索 MakerWorld（如 wall-e, gridfinity, headphone stand…）',
    search_btn: '搜索', mw_url_label: '粘贴 MakerWorld 模型链接', import_btn: '导入',
    attached_label: '当前已绑定参考', no_refs: '暂无绑定参考',
    use_ref_btn: '+ 带入许愿', open_mw_btn: '打开 ↗',
    modal_mobile_title: '📱 手机 App / 同 Wi-Fi 扫码直连',
    copy_btn: '复制', copied_btn: '已复制',
    lan_ok: '✅ 局域网已开放，手机连同一 Wi-Fi 扫码即可许愿',
    lan_local_only: '⚠️ 当前仅本机模式 (127.0.0.1)，加 --lan 启动即可扫码连接',
    lan_hint_ok: (ip) => `手机与电脑连同一 Wi-Fi（IP: ${ip}），手机端对话许愿，算力自动走电脑 GPU。`,
    lan_hint_local: (cmd) => `运行「${cmd}」或打开桌面版即可开放局域网手机扫码。`,
    guide_ios_t: '🍎 iOS:', guide_ios_d: 'Safari 扫码打开 → 分享 →「添加到主屏幕」，全屏运行。',
    guide_and_t: '🤖 Android:', guide_and_d: 'Chrome 扫码打开 → 菜单 →「安装应用 / 添加到主屏幕」。',
    slot_lbl: (s) => `槽${s}`,
  },
  en: {
    brand_tag: 'Wish → 3MF',
    new_btn: '✨ New Wish',
    mw_btn: '🔎 MakerWorld',
    mobile_btn: '📱 Mobile',
    theme_to_dark: '🌙 Midnight',
    theme_to_light: '☀️ Sunlit',
    lang_btn: '中文',
    chat_ph: 'Speak your 3D printing wish to the Spool Genie… (Enter to wish, Ctrl+V image)',
    attach_ref: '📎 Ref / STL',
    mw_chip: '🔎 MakerWorld',
    auto_2d: 'Auto 2D Vision',
    send_btn: '✨ Make Wish',
    lamp_whisper_idle: '✨ Click to rub the Spool Lamp · Awaken the Genie',
    lamp_whispers: [
      '🧞‍♂️ The Spool Genie awakens! Tell me what character or functional print you wish for.',
      '✨ Your multicolor filament spools are humming — ready to manifest onto the build plate!',
      '🔮 Speak your idea & dimensions; I will conjure the 2D vision, joints, and Bambu .3MF.',
    ],
    hero_title: 'Rub the Spool Lamp. Wish It Into Reality.',
    hero_sub: 'Just start chatting your wish. Once your 2D vision appears, printer model, Bambu AMS units, filament swatches, glue-free joints, and slicing options appear right inside the conversation.',
    starters: [
      { title: '📜 Wish · WALL-E Functional Desk Helper', desc: 'Preserve the iconic eyes while carving a rear pen caddy or phone dock', prompt: 'I want a Pixar WALL-E robot with real desktop utility rather than just a figurine—what functional ideas do you suggest?' },
      { title: '📜 Wish · No-AMS Dovetail Stand', desc: 'Matte black & orange split with glue-free sliding dovetails (0g poop)', prompt: 'Design a desktop headphone stand with rear cable slot, 210mm tall, matte black and orange, no AMS, glue-free dovetail split parts' },
      { title: '📜 Wish · Iron Man MagSafe Dock', desc: 'Chibi chest bust with MagSafe charger pocket, 1x AMS 4 colors', prompt: 'Design a stylized Marvel Iron Man chest bust with a MagSafe wireless charger pocket, 125mm tall, 1 AMS unit 4 colors' },
      { title: '📜 Wish · Capybara with Yuzu', desc: 'Cute desktop companion powered by 171 silent built-in priors', prompt: 'Design a cute capybara with a tangerine on its head, 115mm tall, 1 AMS unit 4 colors' },
    ],
    card_2d_title: '🔮 Wish Vision (2D) & Manifestation Setup',
    zoom_btn: '🔍 Zoom',
    sec_printer: '🖨️ 1. Choose Your Printer',
    sec_ams: '🌈 2. Bambu AMS Units & Color Strategy',
    sec_joint: '🧩 3. Glue-Free Mechanical Joint',
    sec_dims_color: '📏 4. Dimensions (mm) & Filament Colors',
    sec_slice: '🔪 5. Export & Slicing',
    ams_opts: [
      { n: 0, t: 'No AMS (0)', s: 'Split · 0g poop' },
      { n: 1, t: '1 AMS', s: '≤ 4 colors' },
      { n: 2, t: '2 AMS', s: '≤ 8 colors' },
      { n: 3, t: '3 AMS', s: '≤ 12 colors' },
      { n: 4, t: '4 AMS', s: '≤ 16 colors' },
    ],
    mode_opts: [
      { v: 'auto', t: 'Auto' },
      { v: 'ams', t: 'AMS Single-Plate' },
      { v: 'split_no_ams', t: 'No-AMS Split (0 Poop)' },
      { v: 'monochrome', t: 'Monochrome' },
    ],
    joint_opts: [
      { v: 'dovetail', t: 'Dovetail', s: 'Strongest' },
      { v: 'snap_pin', t: 'Snap-Pin', s: 'Click fit' },
      { v: 'press_tenon', t: 'Press Tenon', s: 'Flush fit' },
      { v: 'none', t: 'None', s: 'Single piece' },
    ],
    slice_opts: [
      { v: true, t: 'Export .3MF + Auto-Slice', s: 'Uses Bambu/Orca CLI if installed' },
      { v: false, t: 'Export .3MF / .STL / .STEP only', s: 'Slice manually in Studio' },
    ],
    add_color: '+ Add Color',
    swatch_hint: 'Quick Bambu PLA swatches:',
    refresh_2d_btn: '🔄 Update & Re-conjure 2D',
    build_3d_btn: '✨ Manifest 3D & .3MF',
    rendering_2d_inline: '🧞‍♂️ Conjuring your 2D wish vision in background — keep chatting anytime…',
    building_3d_inline: '✨ Manifesting watertight 3D mesh, glue-free joints & Bambu .3MF…',
    thinking_msg: '🧞‍♂️ The Spool Genie is pondering your wish…',
    card_3d_title: '🏆 Wish Manifested · Interactive 3D & Print Files',
    tab_3d: '🧊 3D Model',
    tab_split: '🧩 No-AMS Split',
    tab_tt: '📐 4-Angle Sheet',
    cam_iso: '3/4', cam_front: 'Front', cam_side: 'Side', cam_top: 'Top', cam_wire: 'Wire', cam_spin: 'Spin',
    m_watertight: 'Manifold', m_mass: 'Filament', m_flush: 'AMS Flush', m_time: 'Est. Time',
    wt_ok: '✅ Watertight', wt_warn: '⚠️ Repair',
    dfm_toggle: 'Inspect DFM checks, AMS color decisions & split parts',
    dl_3mf: '⬇ Bambu Multicolor .3MF', dl_3mf_tag: 'Slice Ready',
    dl_split: '⬇ No-AMS Split Plates .3MF', dl_split_tag: '0 Poop',
    dl_stl: '⬇ Universal .STL', dl_stl_tag: 'All Slicers',
    dl_step: '⬇ Parametric CAD .STEP', dl_step_tag: 'Editable',
    dl_py: '⬇ build123d Source .PY', dl_gcode: '⬇ Sliced .GCODE.3MF',
    dl_part: (k) => `⬇ Color Part ${k}.stl`,
    mw_drawer_title: '🔎 MakerWorld Inspiration Vault',
    mw_note: 'Search live MakerWorld models and weave any into your wish, or paste a URL to extract colors & structure.',
    mw_ph: 'Search MakerWorld (e.g. wall-e, gridfinity, headphone stand…)',
    search_btn: 'Search', mw_url_label: 'Paste MakerWorld Model URL', import_btn: 'Import',
    attached_label: 'Bound References', no_refs: 'No references bound',
    use_ref_btn: '+ Weave into Wish', open_mw_btn: 'Open ↗',
    modal_mobile_title: '📱 Mobile App / Same-Wi-Fi QR Pairing',
    copy_btn: 'Copy', copied_btn: 'Copied',
    lan_ok: '✅ LAN Active — scan on iPhone or Android on the same Wi-Fi',
    lan_local_only: '⚠️ Localhost-only mode — start with --lan to enable phone scanning',
    lan_hint_ok: (ip) => `Same Wi-Fi (IP: ${ip}): make wishes on your phone while your computer GPU renders 2D/3D.`,
    lan_hint_local: (cmd) => `Run "${cmd}" or launch the Desktop App to pair phones over Wi-Fi.`,
    guide_ios_t: '🍎 iOS:', guide_ios_d: 'Open in Safari → Share → "Add to Home Screen".',
    guide_and_t: '🤖 Android:', guide_and_d: 'Open in Chrome → Menu → "Install app / Add to Home Screen".',
    slot_lbl: (s) => `S${s}`,
  },
};

function t(key) {
  return (I18N[lang] && I18N[lang][key]) || I18N.zh[key] || key;
}

function setStatus(msg) {
  const el = $('statusInline');
  if (el) el.textContent = msg || '';
}

function spawnSparkles(x, y, count = 11) {
  const layer = $('sparkleLayer');
  if (!layer) return;
  const colors = ['#ffe08a', '#f5b942', '#c4b5fd', '#ff9f5a', '#6ee7b7'];
  for (let i = 0; i < count; i++) {
    const el = document.createElement('div');
    el.className = 'sparkle-star';
    const angle = (Math.PI * 2 * i) / count + (Math.random() - 0.5) * 0.4;
    const dist = 34 + Math.random() * 58;
    el.style.left = `${x}px`;
    el.style.top = `${y}px`;
    el.style.setProperty('--dx', `${Math.cos(angle) * dist}px`);
    el.style.setProperty('--dy', `${Math.sin(angle) * dist - 22}px`);
    const color = colors[i % colors.length];
    el.innerHTML = `<svg viewBox="0 0 24 24" width="14" height="14"><path d="M12 1 L14.5 9.5 L23 12 L14.5 14.5 L12 23 L9.5 14.5 L1 12 L9.5 9.5 Z" fill="${color}"/></svg>`;
    layer.appendChild(el);
    setTimeout(() => el.remove(), 900);
  }
}

function applyTheme(nextTheme) {
  theme = nextTheme === 'light' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', theme);
  localStorage.setItem('sg_theme', theme);
  const metaEl = $('metaThemeColor');
  if (metaEl) metaEl.setAttribute('content', theme === 'dark' ? '#0a0b1a' : '#faf6ee');
  $('themeToggleBtn').textContent = theme === 'dark' ? t('theme_to_light') : t('theme_to_dark');
  if (threeCtx?.scene) {
    threeCtx.scene.background = new THREE.Color(theme === 'dark' ? 0x0c0e22 : 0xf3ede2);
  }
}

function applyLang(nextLang) {
  lang = nextLang === 'en' ? 'en' : 'zh';
  localStorage.setItem('sg_lang', lang);
  document.documentElement.lang = lang === 'en' ? 'en' : 'zh-CN';
  document.querySelectorAll('[data-i18n]').forEach((el) => {
    const val = t(el.dataset.i18n);
    if (typeof val === 'string') el.textContent = val;
  });
  $('langToggleBtn').textContent = t('lang_btn');
  $('themeToggleBtn').textContent = theme === 'dark' ? t('theme_to_light') : t('theme_to_dark');
  $('chatInput').placeholder = t('chat_ph');
  if ($('mwQuery')) $('mwQuery').placeholder = t('mw_ph');
  const firstOpt = $('projectSwitcher')?.options?.[0];
  if (firstOpt && firstOpt.value === '') {
    firstOpt.textContent = lang === 'en' ? '✨ New Wish Sanctuary…' : '✨ 许愿殿堂（新对话）…';
  }
  renderMwQuickTags();
  if (project) renderStream();
}

function scrollChatBottom() {
  const sc = $('chatScroll');
  if (sc) sc.scrollTop = sc.scrollHeight;
}

function openLightbox(src, caption = '') {
  $('lightboxImg').src = src;
  $('lightboxCap').textContent = caption;
  $('lightboxModal').classList.remove('hidden');
}

function destroyThree() {
  if (!threeCtx) return;
  cancelAnimationFrame(threeCtx.rafId);
  if (threeCtx.resizeObs) threeCtx.resizeObs.disconnect();
  threeCtx.renderer?.dispose();
  threeCtx = null;
}

function mountInlineThree(container, glbUrl, hudEls) {
  destroyThree();
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(theme === 'dark' ? 0x0c0e22 : 0xf3ede2);

  const w = Math.max(container.clientWidth, 300);
  const h = Math.max(container.clientHeight, 260);
  const camera = new THREE.PerspectiveCamera(40, w / h, 0.5, 4000);
  camera.position.set(190, 150, 210);

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(w, h);
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.15;
  container.innerHTML = '';
  container.appendChild(renderer.domElement);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.autoRotate = autoSpin;
  controls.autoRotateSpeed = 1.3;

  scene.add(new THREE.HemisphereLight(0xfff7e6, 0x3b3564, 1.5));
  const key = new THREE.DirectionalLight(0xfffaf0, 1.6);
  key.position.set(180, 280, 190);
  scene.add(key);
  const rim = new THREE.DirectionalLight(0xb794f4, 0.9);
  rim.position.set(-190, 130, -170);
  scene.add(rim);

  const pInfo = meta?.printers?.[project?.spec?.printer_id];
  const bedSize = pInfo ? Math.max(pInfo.bed_x_mm, pInfo.bed_y_mm) : 256;
  const grid = new THREE.GridHelper(
    bedSize,
    Math.round(bedSize / 16),
    theme === 'dark' ? 0x6b5b95 : 0xc7a97b,
    theme === 'dark' ? 0x262344 : 0xe3d6c0
  );
  scene.add(grid);

  const ctx = {
    scene, camera, renderer, controls,
    modelGroup: null,
    lastSize: new THREE.Vector3(100, 100, 100),
    rafId: 0,
    resizeObs: null,
  };
  threeCtx = ctx;

  const setCam = (preset) => {
    const maxDim = Math.max(ctx.lastSize.x, ctx.lastSize.y, ctx.lastSize.z, 70);
    const cy = ctx.lastSize.y * 0.45;
    controls.target.set(0, cy, 0);
    if (preset === 'front') camera.position.set(0, cy * 1.05, maxDim * 2.0);
    else if (preset === 'side') camera.position.set(maxDim * 2.0, cy * 1.05, 0);
    else if (preset === 'top') camera.position.set(0, maxDim * 2.2, 0.01);
    else camera.position.set(maxDim * 1.4, maxDim * 1.1, maxDim * 1.45);
    controls.update();
  };
  ctx.setCam = setCam;

  ctx.applyWire = () => {
    if (!ctx.modelGroup) return;
    ctx.modelGroup.traverse((o) => {
      if (o.isMesh && o.material) o.material.wireframe = wireframeOn;
    });
  };

  if (typeof ResizeObserver !== 'undefined') {
    const ro = new ResizeObserver(() => {
      const nw = container.clientWidth;
      const nh = Math.max(container.clientHeight, 1);
      if (nw > 0 && nh > 0) {
        camera.aspect = nw / nh;
        camera.updateProjectionMatrix();
        renderer.setSize(nw, nh);
      }
    });
    ro.observe(container);
    ctx.resizeObs = ro;
  }

  const loop = () => {
    ctx.rafId = requestAnimationFrame(loop);
    controls.autoRotate = autoSpin;
    controls.update();
    renderer.render(scene, camera);
  };
  loop();

  if (glbUrl) {
    gltfLoader.load(
      `${glbUrl}?t=${Date.now()}`,
      (gltf) => {
        if (threeCtx !== ctx) return;
        const wrapper = new THREE.Group();
        gltf.scene.rotation.x = -Math.PI / 2;
        wrapper.add(gltf.scene);
        const box = new THREE.Box3().setFromObject(wrapper);
        const center = box.getCenter(new THREE.Vector3());
        const size = box.getSize(new THREE.Vector3());
        wrapper.position.set(-center.x, -box.min.y, -center.z);
        scene.add(wrapper);
        ctx.modelGroup = wrapper;
        ctx.lastSize = size.clone();
        ctx.applyWire();
        setCam('iso');
        if (hudEls?.dims) hudEls.dims.textContent = `${size.x.toFixed(1)} × ${size.z.toFixed(1)} × ${size.y.toFixed(1)} mm`;
      },
      undefined,
      () => { if (hudEls?.dims) hudEls.dims.textContent = '3D load error'; }
    );
  }
}

function renderAttachedRefs() {
  const strip = $('attachedStrip');
  const drawerList = $('attachedRefsDrawer');
  const refs = project?.references || [];
  if (refs.length) {
    strip.classList.remove('hidden');
    strip.innerHTML = refs.map((r) => `<span class="att-pill">📎 ${r.title}</span>`).join('');
    drawerList.innerHTML = refs.map((r) => `<div class="att-pill">📎 ${r.title} (${r.source})</div>`).join('');
  } else {
    strip.classList.add('hidden');
    strip.innerHTML = '';
    drawerList.innerHTML = `<div>${t('no_refs')}</div>`;
  }
}

function buildSpecPatch(spec) {
  return {
    printer_id: spec.printer_id,
    ams_units: spec.ams_units,
    multicolor_mode: spec.multicolor_mode,
    joint_type: spec.joint_type,
    dimensions: spec.dimensions,
    filaments: spec.filaments,
  };
}

function buildInlineSetupCard(spec, latestConcept) {
  const card = document.createElement('section');
  card.className = 'inline-card';
  const printers = Object.values(meta?.printers || {});
  const showJointSection = spec.ams_units === 0 || spec.multicolor_mode === 'split_no_ams' || spec.joint_type !== 'none';

  card.innerHTML = `
    <div class="icard-head">
      <div class="icard-title">${t('card_2d_title')}</div>
      <span class="badge-soft">${spec.dimensions.width_mm} × ${spec.dimensions.depth_mm} × ${spec.dimensions.height_mm} mm · ${spec.ams_units === 0 ? 'No-AMS' : `${spec.ams_units} AMS`}</span>
    </div>
    <div class="icard-body">
      ${isRendering2D
        ? `<div class="magic-ritual-loader"><div class="magic-orb-spin"></div><span>${t('rendering_2d_inline')}</span></div>`
        : latestConcept?.image_url
          ? `<div class="concept-stage">
               <img src="${latestConcept.image_url}?t=${Date.now()}" alt="2D Vision" class="js-zoom-2d" />
               <div class="concept-bar">
                 <span>✨ ${latestConcept.backend} · ${spec.filaments.length} colors</span>
                 <button type="button" class="btn-link js-zoom-2d">${t('zoom_btn')}</button>
               </div>
             </div>`
          : ''}

      <div class="choice-section">
        <div class="choice-label"><span>${t('sec_printer')}</span></div>
        <div class="pill-row js-printer-pills">
          ${printers.map((p) => `
            <button type="button" class="choice-pill ${p.id === spec.printer_id ? 'active' : ''}" data-printer="${p.id}">
              <span>${p.name.replace('Bambu Lab ', '')}</span>
              <small>${p.bed_x_mm}×${p.bed_y_mm}×${p.bed_z_mm}</small>
            </button>
          `).join('')}
        </div>
      </div>

      <div class="choice-section">
        <div class="choice-label"><span>${t('sec_ams')}</span></div>
        <div class="pill-row js-ams-pills">
          ${t('ams_opts').map((o) => `
            <button type="button" class="choice-pill ${Number(spec.ams_units) === o.n ? 'active' : ''}" data-ams="${o.n}">
              <span>${o.t}</span><small>${o.s}</small>
            </button>
          `).join('')}
        </div>
        <div class="pill-row js-mode-pills">
          ${t('mode_opts').map((m) => `
            <button type="button" class="choice-pill ${spec.multicolor_mode === m.v ? 'active' : ''}" data-mode="${m.v}">
              <span>${m.t}</span>
            </button>
          `).join('')}
        </div>
      </div>

      <div class="choice-section ${showJointSection ? '' : 'hidden'} js-joint-section">
        <div class="choice-label"><span>${t('sec_joint')}</span></div>
        <div class="pill-row js-joint-pills">
          ${t('joint_opts').map((j) => `
            <button type="button" class="choice-pill ${spec.joint_type === j.v ? 'active' : ''}" data-joint="${j.v}">
              <span>${j.t}</span><small>${j.s}</small>
            </button>
          `).join('')}
        </div>
      </div>

      <div class="choice-section">
        <div class="choice-label">
          <span>${t('sec_dims_color')}</span>
          <button type="button" class="btn-link js-add-color">${t('add_color')}</button>
        </div>
        <div class="dims-inline">
          <label class="dim-box">W <input type="number" min="10" max="400" class="js-dim-w" value="${spec.dimensions.width_mm}" /> mm</label>
          <label class="dim-box">D <input type="number" min="10" max="400" class="js-dim-d" value="${spec.dimensions.depth_mm}" /> mm</label>
          <label class="dim-box">H <input type="number" min="4" max="400" class="js-dim-h" value="${spec.dimensions.height_mm}" /> mm</label>
        </div>
        <div class="filament-chips js-filament-chips">
          ${spec.filaments.map((fl, idx) => `
            <div class="fl-chip" data-idx="${idx}">
              <input type="color" value="${fl.hex_color}" class="js-fl-color" />
              <input type="text" value="${fl.name}" class="js-fl-name" />
              <span class="choice-hint">${t('slot_lbl')(fl.slot)}</span>
              ${spec.filaments.length > 1 ? `<button type="button" class="js-fl-del" data-idx="${idx}">×</button>` : ''}
            </div>
          `).join('')}
        </div>
        <div class="choice-hint">${t('swatch_hint')}</div>
        <div class="swatch-strip js-swatches">
          ${Object.entries(meta?.swatches || {}).map(([name, hx]) => `
            <button type="button" class="swatch-dot" style="background:${hx}" title="${name} (${hx})" data-name="${name}" data-hex="${hx}"></button>
          `).join('')}
        </div>
      </div>

      <div class="choice-section">
        <div class="choice-label"><span>${t('sec_slice')}</span></div>
        <div class="pill-row js-slice-pills">
          ${t('slice_opts').map((s) => `
            <button type="button" class="choice-pill ${runSlicerPref === s.v ? 'active' : ''}" data-slice="${s.v}">
              <span>${s.t}</span><small>${s.s}</small>
            </button>
          `).join('')}
        </div>
      </div>

      <div class="icard-actions">
        <button type="button" class="btn btn-secondary js-refresh-2d" ${isBuilding3D ? 'disabled' : ''}>${t('refresh_2d_btn')}</button>
        <div class="icard-actions-right">
          <button type="button" class="btn btn-magic js-build-3d" ${isBuilding3D ? 'disabled' : ''}>${t('build_3d_btn')}</button>
        </div>
      </div>
    </div>
  `;

  card.querySelectorAll('.js-zoom-2d').forEach((el) => {
    el.onclick = () => {
      if (latestConcept?.image_url) {
        openLightbox(`${latestConcept.image_url}?t=${Date.now()}`, `${spec.title} (${spec.dimensions.width_mm}×${spec.dimensions.depth_mm}×${spec.dimensions.height_mm} mm)`);
      }
    };
  });

  const syncInputs = () => {
    spec.dimensions.width_mm = Number(card.querySelector('.js-dim-w').value) || spec.dimensions.width_mm;
    spec.dimensions.depth_mm = Number(card.querySelector('.js-dim-d').value) || spec.dimensions.depth_mm;
    spec.dimensions.height_mm = Number(card.querySelector('.js-dim-h').value) || spec.dimensions.height_mm;
    const fls = [];
    card.querySelectorAll('.fl-chip').forEach((chip, i) => {
      fls.push({
        slot: i + 1,
        hex_color: chip.querySelector('.js-fl-color').value.toUpperCase(),
        name: chip.querySelector('.js-fl-name').value || `Color ${i + 1}`,
        material: 'PLA',
        role: i === 0 ? 'main body' : `accent ${i}`,
      });
    });
    if (fls.length) spec.filaments = fls;
  };

  card.querySelectorAll('.js-printer-pills .choice-pill').forEach((b) => {
    b.onclick = () => { syncInputs(); spec.printer_id = b.dataset.printer; renderStream(); };
  });
  card.querySelectorAll('.js-ams-pills .choice-pill').forEach((b) => {
    b.onclick = () => {
      syncInputs();
      const n = Number(b.dataset.ams);
      spec.ams_units = n;
      if (n === 0) {
        spec.multicolor_mode = 'split_no_ams';
        if (spec.joint_type === 'none') spec.joint_type = 'dovetail';
      } else if (spec.multicolor_mode === 'split_no_ams') {
        spec.multicolor_mode = 'ams';
      }
      renderStream();
    };
  });
  card.querySelectorAll('.js-mode-pills .choice-pill').forEach((b) => {
    b.onclick = () => {
      syncInputs();
      spec.multicolor_mode = b.dataset.mode;
      if (b.dataset.mode === 'split_no_ams' && spec.joint_type === 'none') spec.joint_type = 'dovetail';
      renderStream();
    };
  });
  card.querySelectorAll('.js-joint-pills .choice-pill').forEach((b) => {
    b.onclick = () => { syncInputs(); spec.joint_type = b.dataset.joint; renderStream(); };
  });
  card.querySelectorAll('.js-slice-pills .choice-pill').forEach((b) => {
    b.onclick = () => { syncInputs(); runSlicerPref = b.dataset.slice === 'true'; renderStream(); };
  });
  card.querySelector('.js-add-color').onclick = () => {
    syncInputs();
    if (spec.filaments.length < 16) {
      const defaults = ['#1E4C8A', '#F2EFE9', '#4D9A4E', '#F4D03F', '#C9302C', '#A28BC7'];
      const next = spec.filaments.length + 1;
      spec.filaments.push({ slot: next, name: `Color ${next}`, hex_color: defaults[(next - 1) % defaults.length], material: 'PLA', role: `accent ${next}` });
      renderStream();
    }
  };
  card.querySelectorAll('.js-fl-del').forEach((btn) => {
    btn.onclick = () => {
      syncInputs();
      const idx = Number(btn.dataset.idx);
      if (spec.filaments.length > 1) {
        spec.filaments.splice(idx, 1);
        spec.filaments.forEach((f, i) => (f.slot = i + 1));
        renderStream();
      }
    };
  });
  card.querySelectorAll('.js-swatches .swatch-dot').forEach((sw) => {
    sw.onclick = () => {
      syncInputs();
      if (spec.filaments.length < 16) {
        spec.filaments.push({ slot: spec.filaments.length + 1, name: sw.dataset.name, hex_color: sw.dataset.hex, material: 'PLA', role: `accent ${spec.filaments.length}` });
        renderStream();
      }
    };
  });

  card.querySelector('.js-refresh-2d').onclick = async (e) => {
    syncInputs();
    const rect = e.currentTarget.getBoundingClientRect();
    spawnSparkles(rect.left + rect.width / 2, rect.top + rect.height / 2, 10);
    isRendering2D = true;
    renderStream();
    scrollChatBottom();
    try {
      const r = await fetch(`/api/projects/${project.project_id}/spec`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ patch: buildSpecPatch(spec), re_render_2d: true }),
      });
      project = await r.json();
    } finally {
      isRendering2D = false;
      renderStream();
      scrollChatBottom();
    }
  };

  card.querySelector('.js-build-3d').onclick = async (e) => {
    syncInputs();
    const rect = e.currentTarget.getBoundingClientRect();
    spawnSparkles(rect.left + rect.width / 2, rect.top + rect.height / 2, 16);
    isBuilding3D = true;
    renderStream();
    scrollChatBottom();
    try {
      await fetch(`/api/projects/${project.project_id}/spec`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ patch: buildSpecPatch(spec), re_render_2d: false }),
      });
      const r = await fetch(`/api/projects/${project.project_id}/confirm-build`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ run_slicer: runSlicerPref }),
      });
      project = await r.json();
      await refreshProjectList(project.project_id);
      active3DTab = project.exports?.urls?.split_preview_glb ? 'split' : '3d';
    } finally {
      isBuilding3D = false;
      renderStream();
      scrollChatBottom();
      spawnSparkles(window.innerWidth / 2, window.innerHeight * 0.45, 18);
    }
  };

  return card;
}

function buildInline3DExportCard() {
  const d = project.dfm;
  const exp = project.exports || {};
  const urls = exp.urls || {};
  const card = document.createElement('section');
  card.className = 'inline-card';

  const hasSplit = Boolean(urls.split_preview_glb);
  const hasTT = Boolean(urls.turntable_preview);
  if (active3DTab === 'split' && !hasSplit) active3DTab = '3d';
  if (active3DTab === 'turntable' && !hasTT) active3DTab = '3d';

  const links = [];
  if (urls.threemf) links.push(`<a class="dl-btn primary" href="${urls.threemf}" download>${t('dl_3mf')} <span class="dl-tag">${t('dl_3mf_tag')}</span></a>`);
  if (urls.split_plates_3mf) links.push(`<a class="dl-btn primary" href="${urls.split_plates_3mf}" download>${t('dl_split')} <span class="dl-tag">${t('dl_split_tag')}</span></a>`);
  if (urls.stl) links.push(`<a class="dl-btn" href="${urls.stl}" download>${t('dl_stl')} <span class="dl-tag">${t('dl_stl_tag')}</span></a>`);
  if (urls.step) links.push(`<a class="dl-btn" href="${urls.step}" download>${t('dl_step')} <span class="dl-tag">${t('dl_step_tag')}</span></a>`);
  if (urls.cad_script) links.push(`<a class="dl-btn" href="${urls.cad_script}" download>${t('dl_py')} <span class="dl-tag">Python</span></a>`);
  if (urls.gcode_3mf) links.push(`<a class="dl-btn primary" href="${urls.gcode_3mf}" download>${t('dl_gcode')} <span class="dl-tag">Sliced</span></a>`);
  for (const [k, u] of Object.entries(urls.per_color_stls || {})) {
    if (u) links.push(`<a class="dl-btn" href="${u}" download>${t('dl_part')(k)}</a>`);
  }

  let dfmLines = '';
  for (const c of d?.checks || []) {
    dfmLines += `<div class="check-line ${c.status}"><strong>${c.name}:</strong> ${c.detail}</div>`;
  }
  for (const cd of d?.color_decisions || []) {
    dfmLines += `<div class="choice-hint">• ${cd.source_hex} → ${t('slot_lbl')(cd.mapped_slot)} ${cd.mapped_name} (${cd.action}${cd.delta_e ? `, ΔE=${cd.delta_e}` : ''})</div>`;
  }
  if ((d?.split_parts || []).length > 1) {
    for (const sp of d.split_parts) {
      dfmLines += `<div class="choice-hint">• Plate ${sp.plate_index} [${t('slot_lbl')(sp.slot)}] ${sp.name} (${sp.bbox_mm.join('×')} mm · ${sp.joint_type})</div>`;
    }
  }

  card.innerHTML = `
    <div class="icard-head">
      <div class="icard-title">${t('card_3d_title')}</div>
      <div class="viewer-tabs">
        <button type="button" class="vtab ${active3DTab === '3d' ? 'active' : ''}" data-vtab="3d">${t('tab_3d')}</button>
        ${hasSplit ? `<button type="button" class="vtab ${active3DTab === 'split' ? 'active' : ''}" data-vtab="split">${t('tab_split')}</button>` : ''}
        ${hasTT ? `<button type="button" class="vtab ${active3DTab === 'turntable' ? 'active' : ''}" data-vtab="turntable">${t('tab_tt')}</button>` : ''}
      </div>
    </div>
    <div class="icard-body">
      <div class="three-stage">
        ${active3DTab === 'turntable'
          ? `<img class="turntable-img js-zoom-tt" src="${urls.turntable_preview}?t=${Date.now()}" alt="4-Angle Sheet" />`
          : `<div class="three-canvas js-three-mount"></div>
             <div class="cam-toolbar">
               <button type="button" class="cam-btn" data-cam="iso">${t('cam_iso')}</button>
               <button type="button" class="cam-btn" data-cam="front">${t('cam_front')}</button>
               <button type="button" class="cam-btn" data-cam="side">${t('cam_side')}</button>
               <button type="button" class="cam-btn" data-cam="top">${t('cam_top')}</button>
               <button type="button" class="cam-btn ${wireframeOn ? 'active' : ''} js-wire">${t('cam_wire')}</button>
               <button type="button" class="cam-btn ${autoSpin ? 'active' : ''} js-spin">${t('cam_spin')}</button>
             </div>
             <div class="three-hud">
               <span class="js-hud-dims">${d ? d.bbox_mm.join(' × ') + ' mm' : '-'}</span>
               <span>·</span>
               <span>${d ? `${d.effective_multicolor_mode} (${d.ams_slots_used} col)` : '3D'}</span>
             </div>`}
      </div>

      ${d ? `
        <div class="metrics-grid">
          <div class="metric-box"><strong>${d.watertight ? t('wt_ok') : t('wt_warn')}</strong><span>${t('m_watertight')}</span></div>
          <div class="metric-box"><strong>${exp.estimated_filament_g ?? d.estimated_mass_g} g</strong><span>${t('m_mass')}</span></div>
          <div class="metric-box"><strong>${exp.estimated_flush_g ?? 0} g</strong><span>${t('m_flush')}</span></div>
          <div class="metric-box"><strong>${exp.estimated_print_minutes ? Math.round(exp.estimated_print_minutes) + ' min' : '-'}</strong><span>${t('m_time')}</span></div>
        </div>
        <details class="dfm-details">
          <summary>${t('dfm_toggle')}</summary>
          <div class="dfm-list">${dfmLines}</div>
        </details>
      ` : ''}

      <div class="export-grid">${links.join('')}</div>
    </div>
  `;

  card.querySelectorAll('.vtab').forEach((b) => {
    b.onclick = () => { active3DTab = b.dataset.vtab; renderStream(); };
  });

  const zoomTt = card.querySelector('.js-zoom-tt');
  if (zoomTt) {
    zoomTt.onclick = () => openLightbox(`${urls.turntable_preview}?t=${Date.now()}`, `${project.spec.title} — 4-Angle Sheet`);
  }

  const mountEl = card.querySelector('.js-three-mount');
  if (mountEl) {
    const glbUrl = active3DTab === 'split' ? (urls.split_preview_glb || urls.preview_glb) : urls.preview_glb;
    requestAnimationFrame(() => {
      mountInlineThree(mountEl, glbUrl, { dims: card.querySelector('.js-hud-dims') });
    });
    card.querySelectorAll('.cam-btn[data-cam]').forEach((b) => {
      b.onclick = () => threeCtx?.setCam?.(b.dataset.cam);
    });
    card.querySelector('.js-wire').onclick = (e) => {
      wireframeOn = !wireframeOn;
      e.currentTarget.classList.toggle('active', wireframeOn);
      threeCtx?.applyWire?.();
    };
    card.querySelector('.js-spin').onclick = (e) => {
      autoSpin = !autoSpin;
      e.currentTarget.classList.toggle('active', autoSpin);
    };
  }

  return card;
}

function renderStream() {
  if (!project) return;
  destroyThree();
  renderAttachedRefs();

  const stream = $('chatStream');
  stream.innerHTML = '';

  const hasHistory = (project.history || []).length > 0;
  const latestConcept = project.concepts?.[project.concepts.length - 1];
  const has3D = Boolean(project.exports?.urls?.preview_glb || project.exports?.urls?.threemf);

  if (!hasHistory && !latestConcept && !has3D) {
    const hero = document.createElement('section');
    hero.className = 'welcome-hero';
    hero.innerHTML = `
      <button type="button" class="lamp-orb-btn js-rub-lamp" title="Rub the Spool Lamp!">
        <span class="lamp-ring"></span>
        <span class="lamp-ring-inner"></span>
        <span class="lamp-core">
          <svg viewBox="0 0 24 24" width="34" height="34" fill="none" stroke="#1b1228" stroke-width="2" stroke-linecap="round">
            <circle cx="10.5" cy="13.5" r="7"/><circle cx="10.5" cy="13.5" r="2.3"/>
            <path d="M16.2 8.5 C18.5 6.8 18.8 5.1 17.6 3.8"/>
            <path d="M18.2 1.8 L19.1 4.2 L21.5 5.1 L19.1 6 L18.2 8.4 L17.3 6 L14.9 5.1 L17.3 4.2 Z" fill="#fffdf0" stroke="none"/>
          </svg>
        </span>
      </button>
      <div class="lamp-whisper js-lamp-whisper">${t('lamp_whisper_idle')}</div>
      <h1>${t('hero_title')}</h1>
      <p>${t('hero_sub')}</p>
      <div class="starter-grid">
        ${t('starters').map((s, idx) => `
          <button type="button" class="starter-card" data-sidx="${idx}">
            <strong>${s.title}</strong>
            <span>${s.desc}</span>
          </button>
        `).join('')}
      </div>
    `;
    hero.querySelector('.js-rub-lamp').onclick = (e) => {
      const rect = e.currentTarget.getBoundingClientRect();
      spawnSparkles(rect.left + rect.width / 2, rect.top + rect.height / 2, 16);
      const msgs = t('lamp_whispers');
      hero.querySelector('.js-lamp-whisper').textContent = msgs[lampRubCount % msgs.length];
      lampRubCount += 1;
      $('chatInput').focus();
    };
    hero.querySelectorAll('.starter-card').forEach((btn) => {
      btn.onclick = (e) => {
        const rect = e.currentTarget.getBoundingClientRect();
        spawnSparkles(rect.left + rect.width / 2, rect.top + rect.height / 2, 10);
        sendMessage(t('starters')[Number(btn.dataset.sidx)].prompt);
      };
    });
    stream.appendChild(hero);
    return;
  }

  for (const turn of project.history || []) {
    const div = document.createElement('div');
    div.className = `msg ${turn.role}`;
    div.textContent = turn.content;
    stream.appendChild(div);
  }

  if (latestConcept || isRendering2D || hasHistory) {
    stream.appendChild(buildInlineSetupCard(project.spec, latestConcept));
  }

  if (isBuilding3D) {
    const spin = document.createElement('div');
    spin.className = 'magic-ritual-loader';
    spin.innerHTML = `<div class="magic-orb-spin"></div><span>${t('building_3d_inline')}</span>`;
    stream.appendChild(spin);
  }

  if (has3D && !isBuilding3D) {
    stream.appendChild(buildInline3DExportCard());
  }
}

function startBlankSanctuary() {
  project = {
    project_id: null,
    history: [],
    concepts: [],
    references: [],
    spec: null,
    dfm: null,
    exports: null,
  };
  const sel = $('projectSwitcher');
  if (sel) sel.value = '';
  renderStream();
  setStatus('');
}

async function ensureProject() {
  if (project?.project_id) return project;
  const r = await fetch('/api/projects', { method: 'POST' });
  project = await r.json();
  await refreshProjectList(project.project_id);
  return project;
}

async function refreshProjectList(selectedId = null) {
  const r = await fetch('/api/projects');
  const list = await r.json();
  const sel = $('projectSwitcher');
  sel.innerHTML = '';
  const blankOpt = document.createElement('option');
  blankOpt.value = '';
  blankOpt.textContent = lang === 'en' ? '✨ New Wish Sanctuary…' : '✨ 许愿殿堂（新对话）…';
  sel.appendChild(blankOpt);
  for (const item of list) {
    const opt = document.createElement('option');
    opt.value = item.project_id;
    const shortTitle = (item.title || item.project_id).length > 18
      ? `${item.title.slice(0, 17)}…`
      : (item.title || item.project_id);
    opt.textContent = shortTitle;
    opt.title = `${item.title} (${item.project_id})`;
    sel.appendChild(opt);
  }
  sel.value = selectedId || '';
  return list;
}

async function loadProject(projectId) {
  if (!projectId) {
    startBlankSanctuary();
    return;
  }
  const r = await fetch(`/api/projects/${projectId}`);
  project = await r.json();
  active3DTab = project.exports?.urls?.split_preview_glb ? 'split' : '3d';
  renderStream();
  scrollChatBottom();
  setStatus('');
}

async function createNewProject() {
  startBlankSanctuary();
}

async function sendMessage(text) {
  const clean = (text || '').trim();
  if (!clean || isSendingChat) return;
  await ensureProject();

  isSendingChat = true;
  $('sendBtn').disabled = true;

  const btnRect = $('sendBtn').getBoundingClientRect();
  spawnSparkles(btnRect.left + btnRect.width / 2, btnRect.top + btnRect.height / 2, 10);

  const stream = $('chatStream');
  const hero = stream.querySelector('.welcome-hero');
  if (hero) hero.remove();

  const uDiv = document.createElement('div');
  uDiv.className = 'msg user';
  uDiv.textContent = clean;
  stream.appendChild(uDiv);

  const tDiv = document.createElement('div');
  tDiv.className = 'msg assistant typing';
  tDiv.textContent = t('thinking_msg');
  stream.appendChild(tDiv);
  scrollChatBottom();

  try {
    const r = await fetch(`/api/projects/${project.project_id}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: clean, auto_render_2d: false }),
    });
    if (!r.ok) throw new Error(`HTTP ${r.status}: ${await r.text()}`);
    const data = await r.json();
    project = data.project;
    await refreshProjectList(project.project_id);

    const want2d = $('auto2dCheck')?.checked ?? true;
    isSendingChat = false;
    $('sendBtn').disabled = false;

    if (want2d) {
      isRendering2D = true;
      renderStream();
      scrollChatBottom();
      try {
        const r2 = await fetch(`/api/projects/${project.project_id}/render-2d`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({}),
        });
        if (r2.ok) project = await r2.json();
      } finally {
        isRendering2D = false;
        renderStream();
        scrollChatBottom();
        spawnSparkles(window.innerWidth / 2, window.innerHeight * 0.4, 12);
      }
    } else {
      renderStream();
      scrollChatBottom();
    }
  } catch (err) {
    tDiv.className = 'msg error';
    tDiv.textContent = `Error: ${err.message || err}`;
    $('chatInput').value = clean;
    isSendingChat = false;
    $('sendBtn').disabled = false;
  }
}

function renderMwQuickTags() {
  const box = $('mwQuickTags');
  if (!box) return;
  const tags = ['WALL-E', 'Gridfinity', 'Headphone Stand', 'Skadis', 'Iron Man', 'Capybara'];
  box.innerHTML = '';
  for (const tag of tags) {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'chip';
    b.textContent = tag;
    b.onclick = () => { $('mwQuery').value = tag; runMakerWorldSearch(tag); };
    box.appendChild(b);
  }
}

async function runMakerWorldSearch(q) {
  const query = (q || '').trim();
  if (!query) return;
  $('mwResults').innerHTML = `<div class="choice-hint">${lang === 'en' ? 'Searching MakerWorld…' : '正在搜索 MakerWorld…'}</div>`;
  try {
    const r = await fetch(`/api/makerworld/search?q=${encodeURIComponent(query)}&limit=8`);
    const items = await r.json();
    $('mwResults').innerHTML = '';
    for (const it of items) {
      const card = document.createElement('div');
      card.className = 'mw-card';
      const thumb = it.thumbnail_url ? `<img class="mw-thumb" src="${it.thumbnail_url}" alt="" onerror="this.style.display='none'" />` : '';
      card.innerHTML = `
        ${thumb}
        <div class="mw-body">
          <a class="mw-title" href="${it.url}" target="_blank" rel="noopener" title="${it.title}">${it.title}</a>
          <div class="mw-snip">${it.snippet || ''}</div>
          <div class="mw-actions">
            <a class="btn-link" href="${it.url}" target="_blank" rel="noopener">${t('open_mw_btn')}</a>
            <button type="button" class="btn-link js-mw-use">${t('use_ref_btn')}</button>
          </div>
        </div>
      `;
      card.querySelector('.js-mw-use').onclick = async () => {
        await ensureProject();
        const resp = await fetch(`/api/projects/${project.project_id}/references/url`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ url: it.url }),
        });
        const data = await resp.json();
        project = data.project;
        renderAttachedRefs();
        $('mwDrawerBackdrop').classList.add('hidden');
        setStatus(lang === 'en' ? `Bound: ${it.title}` : `已绑定：${it.title}`);
      };
      $('mwResults').appendChild(card);
    }
  } catch (err) {
    $('mwResults').innerHTML = `<div class="choice-hint">Error: ${err.message}</div>`;
  }
}

async function uploadReferenceBlob(fileOrBlob, filename) {
  await ensureProject();
  setStatus(lang === 'en' ? `Analyzing ${filename}…` : `正在解析 ${filename}…`);
  const fd = new FormData();
  fd.append('file', fileOrBlob, filename);
  const r = await fetch(`/api/projects/${project.project_id}/references/upload`, { method: 'POST', body: fd });
  const data = await r.json();
  project = data.project;
  renderAttachedRefs();
  setStatus(lang === 'en' ? `Attached ${filename}` : `已绑定参考 ${filename}`);
}

async function openMobileModal() {
  $('mobileModal').classList.remove('hidden');
  try {
    const r = await fetch('/api/lan-info');
    const info = await r.json();
    const pill = $('lanStatusPill');
    pill.textContent = info.lan_enabled ? t('lan_ok') : t('lan_local_only');
    pill.classList.toggle('warn', !info.lan_enabled);
    $('lanQrBox').innerHTML = info.qr_svg || '<div>QR unavailable</div>';
    $('lanUrlInput').value = info.lan_url;
    $('lanHintBox').textContent = info.lan_enabled ? t('lan_hint_ok')(info.primary_ip) : t('lan_hint_local')(info.cli_hint);
  } catch (err) {
    $('lanStatusPill').textContent = `Error: ${err.message || err}`;
  }
}

async function bootstrap() {
  const r = await fetch('/api/meta');
  meta = await r.json();
  const d = meta.doctor;
  const gpuShort = d.gpu_name ? d.gpu_name.replace('NVIDIA GeForce ', '').replace(' Laptop GPU', '') : d.accelerator;
  $('hwBadge').textContent = `⚡ ${gpuShort}`;

  applyTheme(theme);
  applyLang(lang);

  // Start on a clean Wish Lamp Sanctuary if user clicked fresh, or load latest project in switcher
  await refreshProjectList();
  await createNewProject();
}

$('newProjectBtn').onclick = (e) => {
  const rect = e.currentTarget.getBoundingClientRect();
  spawnSparkles(rect.left + rect.width / 2, rect.top + rect.height / 2, 10);
  createNewProject();
};
$('projectSwitcher').onchange = (e) => loadProject(e.target.value);
$('themeToggleBtn').onclick = () => applyTheme(theme === 'dark' ? 'light' : 'dark');
$('langToggleBtn').onclick = () => applyLang(lang === 'zh' ? 'en' : 'zh');

const openMw = () => $('mwDrawerBackdrop').classList.remove('hidden');
$('openMwDrawerBtn').onclick = openMw;
$('composerMwBtn').onclick = openMw;
$('closeMwDrawerBtn').onclick = () => $('mwDrawerBackdrop').classList.add('hidden');
$('mwDrawerBackdrop').addEventListener('click', (e) => {
  if (e.target === $('mwDrawerBackdrop')) $('mwDrawerBackdrop').classList.add('hidden');
});
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
  await ensureProject();
  const r = await fetch(`/api/projects/${project.project_id}/references/url`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ url }),
  });
  const data = await r.json();
  project = data.project;
  $('mwUrlInput').value = '';
  renderAttachedRefs();
  setStatus(lang === 'en' ? `Imported ${data.reference.title}` : `已导入 ${data.reference.title}`);
};

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

$('closeLightboxBtn').onclick = () => $('lightboxModal').classList.add('hidden');
$('lightboxModal').addEventListener('click', (e) => {
  if (e.target === $('lightboxModal')) $('lightboxModal').classList.add('hidden');
});

function submitFromBox() {
  const el = $('chatInput');
  const txt = el.value.trim();
  if (!txt || isSendingChat) return;
  el.value = '';
  el.style.height = 'auto';
  sendMessage(txt);
}

$('chatForm').addEventListener('submit', (e) => { e.preventDefault(); submitFromBox(); });
$('chatInput').addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey && !e.isComposing && e.keyCode !== 229) {
    e.preventDefault();
    submitFromBox();
  }
});
$('chatInput').addEventListener('input', (e) => {
  e.target.style.height = 'auto';
  e.target.style.height = `${Math.min(e.target.scrollHeight, 140)}px`;
});
$('chatInput').addEventListener('paste', async (e) => {
  const items = e.clipboardData?.items || [];
  for (const item of items) {
    if (item.type && item.type.startsWith('image/')) {
      e.preventDefault();
      const blob = item.getAsFile();
      if (blob) await uploadReferenceBlob(blob, `pasted_ref_${Date.now()}.png`);
      return;
    }
  }
});
$('refUpload').onchange = async (e) => {
  const file = e.target.files?.[0];
  if (!file) return;
  await uploadReferenceBlob(file, file.name);
  e.target.value = '';
};

if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('/sw.js').catch(() => {});
}

bootstrap();
