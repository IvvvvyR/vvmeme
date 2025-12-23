import os
import json
import random
import asyncio
import time
import re
import aiohttp
import difflib
import zipfile
import io
import datetime
from concurrent.futures import ThreadPoolExecutor
from aiohttp import web
from PIL import Image as PILImage

# 尝试导入 lunar_python，没有也不报错
try:
    from lunar_python import Lunar, Solar
    HAS_LUNAR = True
except ImportError:
    HAS_LUNAR = False
    print("【提示】未安装 lunar_python，将只显示阳历时间。建议 pip install lunar_python")

from astrbot.api.star import Context, Star, register
from astrbot.api.event import filter, AstrMessageEvent, MessageChain
from astrbot.api.event.filter import EventMessageType
from astrbot.core.message.components import Image, Plain

print("DEBUG: MemeMaster Pro (v1.6.1 Final) 已加载")

@register("vv_meme_master", "MemeMaster", "全功能完整版", "1.6.1")
class MemeMaster(Star):
    def __init__(self, context: Context, config: dict = None):
        super().__init__(context)
        self.base_dir = os.path.abspath(os.path.dirname(__file__))
        self.img_dir = os.path.join(self.base_dir, "images")
        self.data_file = os.path.join(self.base_dir, "memes.json")
        self.config_file = os.path.join(self.base_dir, "config.json")
        self.memory_file = os.path.join(self.base_dir, "memory.txt")
        
        self.executor = ThreadPoolExecutor(max_workers=3)
        if not os.path.exists(self.img_dir): os.makedirs(self.img_dir, exist_ok=True)
            
        self.local_config = self.load_config()
        self.data = self.load_data()
        self.img_hashes = {}
        
        # 运行时状态管理
        self.debounce_tasks = {}
        self.msg_buffers = {}
        self.chat_history_buffer = [] 
        self.last_active_time = time.time()
        self.current_summary = self.load_memory()
        
        self.pair_map = {'“': '”', '《': '》', '（': '）', '(': ')', '[': ']', '{': '}'}

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.start_web_server())
            loop.create_task(self._init_image_hashes())
            loop.create_task(self._lonely_watcher())
        except Exception as e:
            print(f"ERROR: 启动任务失败: {e}")

    # ==========================
    # 模块 1: 时间系统 (离线版)
    # ==========================
    def get_time_str(self):
        now = datetime.datetime.now()
        week_days = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
        solar_str = f"{now.strftime('%Y年%m月%d日 %H:%M')} {week_days[now.weekday()]}"
        
        lunar_str = ""
        if HAS_LUNAR:
            try:
                solar = Solar.fromYmdHms(now.year, now.month, now.day, now.hour, now.minute, now.second)
                lunar = solar.getLunar()
                jieqi = lunar.getJieQi()
                lunar_str = f" 农历{lunar.getMonthInChinese()}月{lunar.getDayInChinese()}"
                if jieqi: lunar_str += f" ({jieqi})"
            except: pass
            
        return f"[当前时间: {solar_str}{lunar_str}]"

    # ==========================
    # 替换模块 2: 寂寞主动聊 (带静默时段)
    # ==========================
    async def _lonely_watcher(self):
        print("[Meme] 寂寞检测启动...")
        while True:
            await asyncio.sleep(60) 
            interval = self.local_config.get("proactive_interval", 0)
            if interval <= 0: continue
            
            # 【新增】静默时段检查
            q_start = self.local_config.get("quiet_start", -1) # -1表示不启用
            q_end = self.local_config.get("quiet_end", -1)
            
            if q_start != -1 and q_end != -1:
                current_hour = datetime.datetime.now().hour
                # 逻辑：比如 23点到7点。
                # 如果 start > end (跨夜)：hour >= 23 或 hour < 7 都是静默
                # 如果 start < end (白天)：hour >= 14 且 hour < 16 是静默
                is_quiet = False
                if q_start > q_end: 
                    if current_hour >= q_start or current_hour < q_end: is_quiet = True
                else:
                    if q_start <= current_hour < q_end: is_quiet = True
                
                if is_quiet:
                    # print(f"[Meme] 当前是静默时段 ({current_hour}点)，跳过主动发送")
                    continue

            if time.time() - self.last_active_time > (interval * 60):
                self.last_active_time = time.time() 
                
                provider = self.context.get_using_provider()
                if provider:
                    # 读取完整的长记忆给 AI 参考
                    full_memory = self.load_memory()
                    ctx = f"{self.get_time_str()}\n你已经很久({interval}分钟)没有和用户说话了。\n[你的长期记忆]: {full_memory}\n请根据记忆和时间，主动发起一个不生硬的话题。"
                    try:
                        sid = getattr(self, "last_session_id", None)
                        resp = await provider.text_chat(ctx, session_id=sid)
                        text = (getattr(resp, "completion_text", None) or getattr(resp, "text", "")).strip()
                        if text:
                            uid = getattr(self, "last_uid", None)
                            if uid: await self.process_and_send(None, text, target_uid=uid)
                    except: pass


    # ==========================
    # 模块 3: 核心消息处理 (防抖+过滤)
    # ==========================
    @filter.event_message_type(EventMessageType.PRIVATE_MESSAGE, priority=50)
    async def handle_private_msg(self, event: AstrMessageEvent):
        # 1. 过滤自己发的消息
        try:
            if str(event.message_obj.sender.user_id) == str(self.context.get_current_provider_bot().self_id): return
        except: pass

        msg_str = (event.message_str or "").strip()
        img_url = self._get_img_url(event)
        uid = event.unified_msg_origin

        # 【核心】强力过滤空消息（屏蔽 NapCat 的输入状态）
        if not msg_str and not img_url: return

        # 更新活跃状态
        self.last_active_time = time.time()
        self.last_session_id = event.session_id
        self.last_uid = uid

        # 记录用户消息用于总结
        if msg_str: self.chat_history_buffer.append(f"User: {msg_str}")

        # 暗线：自动进货
        if img_url and not msg_str.startswith("/"):
            if time.time() - getattr(self, "last_auto_save_time", 0) > self.local_config.get("auto_save_cooldown", 60):
                asyncio.create_task(self.ai_evaluate_image(img_url, msg_str))

        # 指令穿透：直接执行
        if msg_str.startswith(("/", "！", "!")):
            if uid in self.debounce_tasks: 
                self.debounce_tasks[uid].cancel()
                await self._execute_buffer(uid, event)
            return

        debounce_time = self.local_config.get("debounce_time", 5.0)
        if debounce_time <= 0: return

        # 拦截！
        event.stop_event()

        # 存入缓存
        if uid not in self.msg_buffers: self.msg_buffers[uid] = {'text': [], 'imgs': [], 'event': event}
        self.msg_buffers[uid]['event'] = event 
        if msg_str: self.msg_buffers[uid]['text'].append(msg_str)
        if img_url: self.msg_buffers[uid]['imgs'].append(img_url)

        # 续杯：如果有旧计时器，杀掉它
        if uid in self.debounce_tasks and not self.debounce_tasks[uid].done():
            self.debounce_tasks[uid].cancel()

        # 启动新计时器
        self.debounce_tasks[uid] = asyncio.create_task(self._debounce_waiter(uid, debounce_time))

    async def _debounce_waiter(self, uid, duration):
        try:
            await asyncio.sleep(duration)
            await self._execute_buffer(uid)
        except asyncio.CancelledError: pass

    async def _execute_buffer(self, uid, force_event=None):
        if uid not in self.msg_buffers: return
        data = self.msg_buffers.pop(uid)
        event = force_event or data['event']
        texts = data['text']
        imgs = data['imgs']
        if not texts and not imgs: return
        
        print(f"[Meme] 结算: {len(texts)}文本, {len(imgs)}图片")

        # 准备 Prompt
        image_urls = [url for url in imgs]
        user_input = "\n".join(texts)
        
        # 触发记忆总结
        asyncio.create_task(self.check_and_summarize())

        time_info = self.get_time_str()
        memory_info = f"\n[前情提要: {self.current_summary}]" if self.current_summary else ""
        
        # 小抄 (尖括号格式)
        hint_msg = ""
        if random.randint(1, 100) <= self.local_config.get("reply_prob", 50):
            all_tags = [v.get("tags", "").split(":")[0].strip() for v in self.data.values()]
            if all_tags:
                hints = random.sample(all_tags, min(15, len(all_tags)))
                hint_str = " ".join([f"<MEME:{h}>" for h in hints])
                hint_msg = f"\n[可用表情包: {hint_str}]\n回复格式: <MEME:名称>"

        full_prompt = f"{time_info}{memory_info}\n{user_input}{hint_msg}"
        
        # 修改 event 内容 (虽然这里我们自己调 API，但改了也没坏处)
        event.message_str = full_prompt
        
        print(f"[Meme] 请求LLM...")
        provider = self.context.get_using_provider()
        if provider:
            try:
                resp = await provider.text_chat(text=full_prompt, session_id=event.session_id, image_urls=image_urls)
                reply = (getattr(resp, "completion_text", None) or getattr(resp, "text", "")).strip()
                
                if reply:
                    self.chat_history_buffer.append(f"AI: {reply}")
                    await self.process_and_send(event, reply)
            except Exception as e:
                print(f"LLM请求失败: {e}")

    # ==========================
    # 模块 4: 外挂记忆
    # ==========================
    def load_memory(self):
        if os.path.exists(self.memory_file):
            try: return open(self.memory_file, "r", encoding="utf-8").read()
            except: return ""
        return ""

    # ==========================
    # 替换模块 4: 外挂记忆 (改为追加模式)
    # ==========================
    # load_memory 不用改，它是读取整个文件

    async def check_and_summarize(self):
        threshold = self.local_config.get("summary_threshold", 50) 
        if len(self.chat_history_buffer) >= threshold:
            history_text = "\n".join(self.chat_history_buffer)
            self.chat_history_buffer = [] 
            
            print("[Meme] 触发记忆追加...")
            provider = self.context.get_using_provider()
            if provider:
                now_str = self.get_time_str()
                # 【修改点】让 AI 总结这一小段，而不是重写整个记忆
                prompt = f"""当前时间：{now_str}
这是用户和AI最近的{threshold}句对话。
请将这段对话浓缩成一段“日记”，记录关键事件、用户观点或梗，以及恋爱日常。
不要回顾太久远的历史，只总结这段对话。
要求：简洁、带时间标记。

对话内容：
{history_text}"""

                try:
                    resp = await provider.text_chat(prompt, session_id=None)
                    summary = (getattr(resp, "completion_text", None) or getattr(resp, "text", "")).strip()
                    if summary:
                        # 【核心修改】使用 'a' (append) 模式追加，而不是覆盖！
                        with open(self.memory_file, "a", encoding="utf-8") as f: 
                            f.write(f"\n\n--- 记录时间: {now_str} ---\n{summary}")
                        
                        # 更新一下内存里的当前记忆，供下次对话使用
                        self.current_summary = self.load_memory()
                        print(f"[Meme] 记忆已追加: {summary[:20]}...")
                except: pass

    # ==========================
    # 模块 5: 回复处理 (分段+正则+GIF)
    # ==========================
    async def process_and_send(self, event, text, target_uid=None):
        print(f"[Meme] AI回复: {text[:30]}...")
        try:
            # 正则匹配 <MEME:Tag> 或 MEME_TAG:Tag (兼容旧版)
            pattern = r"(<MEME:.*?>|MEME_TAG:\s*[\S]+)"
            parts = re.split(pattern, text)
            mixed_chain = []
            
            for part in parts:
                tag = None
                if part.startswith("<MEME:"): tag = part[6:-1].strip()
                elif "MEME_TAG:" in part: tag = part.replace("MEME_TAG:", "").strip()
                
                if tag:
                    path = self.find_best_match(tag)
                    if path: mixed_chain.append(Image.fromFileSystem(path))
                elif part:
                    # 过滤冒号废话
                    if part.strip().startswith(":") and len(part) < 30: continue
                    mixed_chain.append(Plain(part))
            
            segments = self.smart_split(mixed_chain)
            uid = target_uid or event.unified_msg_origin
            
            delay_base = self.local_config.get("delay_base", 0.5)
            delay_factor = self.local_config.get("delay_factor", 0.1)
            
            for i, seg in enumerate(segments):
                txt_c = "".join([c.text for c in seg if isinstance(c, Plain)])
                mc = MessageChain(); mc.chain = seg
                await self.context.send_message(uid, mc)
                if i < len(segments) - 1:
                    await asyncio.sleep(delay_base + len(txt_c) * delay_factor)
        except Exception as e:
            print(f"发送出错: {e}")

    # ==========================
    # 模块 6: 自动进货 (WebUI Prompt)
    # ==========================
    async def ai_evaluate_image(self, img_url, context_text=""):
        try:
            self.last_auto_save_time = time.time()
            img_data = await self.download_image(img_url)
            if not img_data: return

            loop = asyncio.get_running_loop()
            current_hash = await loop.run_in_executor(self.executor, self.calc_dhash, img_data)
            if current_hash and self.is_duplicate(current_hash): return

            provider = self.context.get_using_provider()
            if not provider: return
            
            # 默认 Prompt
            default_prompt = """你正在整理表情包。用户配文：“{context_text}”。
规则：
1. 二次元/Meme环境，严禁幻觉。黑名单：米哈游/原神/孙笑川/辱女。
2. 若保存，格式：YES\n<MEME:名称>: 简短说明"""
            
            # 从配置读取
            prompt_template = self.local_config.get("ai_prompt", default_prompt)
            prompt = prompt_template.replace("{context_text}", context_text)

            resp = await provider.text_chat(prompt, session_id=None, image_urls=[img_url])
            content = (getattr(resp, "completion_text", None) or getattr(resp, "text", "")).strip()
            
            if "YES" in content:
                match = re.search(r"<MEME:(.*?)>[:：]?(.*)", content)
                if match:
                    full_tag = f"{match.group(1).strip()}: {match.group(2).strip()}"
                    print(f"🖤 [自动进货] {full_tag}")
                    
                    comp_data, ext = await loop.run_in_executor(self.executor, self.compress_image_sync, img_data)
                    fn = f"{int(time.time())}{ext}"
                    with open(os.path.join(self.img_dir, fn), "wb") as f: f.write(comp_data)
                    self.data[fn] = {"tags": full_tag, "source": "auto"}
                    if current_hash: self.img_hashes[fn] = current_hash 
                    self.save_data()
        except: pass

    # ==========================
    # 辅助工具 (GIF支持 + 指纹)
    # ==========================
    def compress_image_sync(self, image_data: bytes) -> tuple[bytes, str]:
        try:
            img = PILImage.open(io.BytesIO(image_data))
            # 【GIF 支持】
            if getattr(img, 'is_animated', False) or img.format == 'GIF': 
                return image_data, ".gif"
            
            max_size = 350 
            w, h = img.size
            if w > max_size or h > max_size:
                if w > h: new_w = max_size; new_h = int(h * (max_size / w))
                else: new_h = max_size; new_w = int(w * (max_size / h))
                img = img.resize((new_w, new_h), PILImage.Resampling.LANCZOS)
            
            buffer = io.BytesIO()
            if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
                if img.mode != "RGBA": img = img.convert("RGBA")
                img.save(buffer, format="PNG", optimize=True); return buffer.getvalue(), ".png"
            else:
                if img.mode != "RGB": img = img.convert("RGB")
                img.save(buffer, format="JPEG", quality=70, optimize=True); return buffer.getvalue(), ".jpg"
        except: return image_data, ".jpg"

    async def _init_image_hashes(self):
        loop = asyncio.get_running_loop()
        for f in os.listdir(self.img_dir):
            if not f.lower().endswith(('.jpg', '.png', '.jpeg', '.gif', '.webp')): continue
            try:
                with open(os.path.join(self.img_dir, f), "rb") as fl: 
                    h = await loop.run_in_executor(self.executor, self.calc_dhash, fl.read())
                    if h: self.img_hashes[f] = h
            except: pass

    def calc_dhash(self, image_data: bytes) -> str:
        try:
            img = PILImage.open(io.BytesIO(image_data))
            if getattr(img, 'is_animated', False): img.seek(0)
            img = img.resize((9, 8), PILImage.Resampling.LANCZOS).convert('L')
            pixels = list(img.getdata()); diff = []
            for row in range(8):
                for col in range(8): diff.append(pixels[row*9+col] > pixels[row*9+col+1])
            val = 0
            for i, v in enumerate(diff): 
                if v: val += 2**i
            return hex(val)[2:]
        except: return None
    
    def is_duplicate(self, h, t=5):
        if not h: return False
        for _, eh in self.img_hashes.items():
            try:
                if bin(int(h, 16) ^ int(eh, 16)).count('1') <= t: return True
            except: continue
        return False

    async def download_image(self, url):
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=8)) as s:
                async with s.get(url) as r: return await r.read() if r.status==200 else None
        except: return None

    def smart_split(self, chain):
        segs = []; buf = []
        for c in chain:
            if isinstance(c, Image): 
                if buf: segs.append(buf[:]); buf.clear()
                segs.append([c]); continue
            if isinstance(c, Plain):
                txt = c.text; idx = 0; chunk = ""
                while idx < len(txt):
                    char = txt[idx]
                    if char in "\n。？！?!":
                        chunk += char
                        if chunk.strip(): buf.append(Plain(chunk))
                        if buf: segs.append(buf[:]); buf.clear()
                        chunk = ""
                    else: chunk += char
                    idx += 1
                if chunk: buf.append(Plain(chunk))
        if buf: segs.append(buf)
        return segs

    def find_best_match(self, query):
        best, score = None, 0
        for f, i in self.data.items():
            t_key = i.get("tags", "").split(":")[0].strip()
            if query == t_key: return os.path.join(self.img_dir, f)
            if query in i.get("tags", ""): return os.path.join(self.img_dir, f)
            s = difflib.SequenceMatcher(None, query, t_key).ratio()
            if s > score: score = s; best = f
        return os.path.join(self.img_dir, best) if score > 0.4 else None

    def _get_img_url(self, e):
        for c in e.message_obj.message:
            if isinstance(c, Image): return c.url
        return None
    def load_config(self): return {**{"web_port":5000,"debounce_time":5.0,"reply_prob":50,"proactive_interval":0,"summary_threshold":50}, **(json.load(open(self.config_file)) if os.path.exists(self.config_file) else {})}
    def save_config(self): json.dump(self.local_config, open(self.config_file,"w"), indent=2)
    def load_data(self): return json.load(open(self.data_file)) if os.path.exists(self.data_file) else {}
    def save_data(self): json.dump(self.data, open(self.data_file,"w"), ensure_ascii=False)

    # ==========================
    # Web Server (全功能)
    # ==========================
    async def start_web_server(self):
        app = web.Application(); app._client_max_size = 50*1024*1024
        app.router.add_get("/", self.h_idx); app.router.add_post("/upload", self.h_up)
        app.router.add_post("/batch_delete", self.h_del); app.router.add_post("/update_tag", self.h_tag)
        app.router.add_get("/get_config", self.h_gcf); app.router.add_post("/update_config", self.h_ucf)
        app.router.add_get("/backup", self.h_backup); app.router.add_post("/restore", self.h_restore)
        app.router.add_post("/slim_images", self.h_slim); app.router.add_static("/images/", path=self.img_dir)
        runner = web.AppRunner(app); await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", self.local_config.get("web_port", 5000))
        await site.start(); print(f"WebUI started")
    
    async def h_idx(self,r): return web.Response(text=self.read_file("index.html").replace("{{MEME_DATA}}", json.dumps(self.data)), content_type="text/html")
    async def h_up(self,r): 
        rd=await r.multipart(); t="未分类"
        while True:
            p=await rd.next()
            if not p: break
            if p.name=="file":
                d=await p.read()
                l=asyncio.get_running_loop()
                h=await l.run_in_executor(self.executor, self.calc_dhash, d)
                c,e=await l.run_in_executor(self.executor, self.compress_image_sync, d)
                fn=f"{int(time.time()*1000)}_{random.randint(100,999)}{e}"
                with open(os.path.join(self.img_dir, fn),"wb") as f: f.write(c)
                self.data[fn]={"tags":t,"source":"manual"}
                if h: self.img_hashes[fn]=h
            elif p.name=="tags": t=await p.text()
        self.save_data(); return web.Response(text="ok")
    async def h_del(self,r):
        for f in (await r.json()).get("filenames",[]):
            try: os.remove(os.path.join(self.img_dir,f)); del self.data[f]
            except: pass
        self.save_data(); return web.Response(text="ok")
    async def h_tag(self,r): d=await r.json(); self.data[d['filename']]['tags']=d['tags']; self.save_data(); return web.Response(text="ok")
    async def h_gcf(self,r): return web.json_response(self.local_config)
    async def h_ucf(self,r): self.local_config.update(await r.json()); self.save_config(); return web.Response(text="ok")
    async def h_backup(self,r):
        b=io.BytesIO()
        with zipfile.ZipFile(b,'w',zipfile.ZIP_DEFLATED) as z:
            for root,_,files in os.walk(self.img_dir): 
                for f in files: z.write(os.path.join(root,f),f"images/{f}")
            if os.path.exists(self.data_file): z.write(self.data_file,"memes.json")
        b.seek(0); return web.Response(body=b, headers={'Content-Disposition':'attachment; filename="bk.zip"'})
    async def h_restore(self,r):
        rd=await r.multipart(); f=await rd.next()
        if not f: return web.Response(status=400)
        try: 
            with zipfile.ZipFile(io.BytesIO(await f.read())) as z: z.extractall(self.base_dir)
            self.data=self.load_data(); self.local_config=self.load_config()
            asyncio.create_task(self._init_image_hashes())
            return web.Response(text="ok")
        except: return web.Response(status=500)
    async def h_slim(self,r):
        count = 0; loop = asyncio.get_running_loop()
        self.img_hashes = {}
        for f in os.listdir(self.img_dir):
            path = os.path.join(self.img_dir, f)
            try:
                with open(path, 'rb') as file: raw = file.read()
                old_size = len(raw)
                h = await loop.run_in_executor(self.executor, self.calc_dhash, raw)
                if h: self.img_hashes[f] = h
                new_data, ext = await loop.run_in_executor(self.executor, self.compress_image_sync, raw)
                if len(new_data) < old_size:
                    with open(path, 'wb') as file: file.write(new_data)
                    count += 1
            except: pass
        return web.Response(text=f"优化 {count} 张")
    
    def read_file(self, n): return open(os.path.join(self.base_dir, n), "r", encoding="utf-8").read()
