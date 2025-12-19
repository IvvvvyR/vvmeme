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
import shutil
from aiohttp import web
from PIL import Image as PILImage  # 引入图片处理库

from astrbot.api.star import Context, Star, register
from astrbot.api.event import filter, AstrMessageEvent, MessageChain
from astrbot.api.event.filter import EventMessageType
from astrbot.core.message.components import Image, Plain

print("DEBUG: MemeMaster Pro (Lioren Fixed) 已加载")

@register("vv_meme_master", "MemeMaster", "防抖+表情包+拟人分段+图片压缩", "1.0.2")
class MemeMaster(Star):
    def __init__(self, context: Context, config: dict = None):
        super().__init__(context)
        self.base_dir = os.path.abspath(os.path.dirname(__file__))
        self.img_dir = os.path.join(self.base_dir, "images")
        self.data_file = os.path.join(self.base_dir, "memes.json")
        self.config_file = os.path.join(self.base_dir, "config.json")
        
        if not os.path.exists(self.img_dir): os.makedirs(self.img_dir, exist_ok=True)
            
        self.local_config = self.load_config()
        self.data = self.load_data()
        
        self.sessions = {}
        # 补充更多成对符号，防止切割错误
        self.pair_map = {'“': '”', '《': '》', '（': '）', '(': ')', '[': ']', '{': '}', '【': '】'}

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.start_web_server())
        except Exception as e:
            print(f"ERROR: Web后台启动失败: {e}")

    # ==========================
    # 核心 1: 输入端防抖 (修复循环BUG)
    # ==========================
    async def _timer_coroutine(self, uid: str, duration: float):
        try:
            await asyncio.sleep(duration)
            if uid in self.sessions: self.sessions[uid]['flush_event'].set()
        except asyncio.CancelledError: pass

    @filter.event_message_type(EventMessageType.PRIVATE_MESSAGE, priority=50)
    async def handle_private_msg(self, event: AstrMessageEvent):
        try:
            msg_str = (event.message_str or "").strip()
            # 过滤掉 System 提示词，防止自循环
            if "[System]" in msg_str: return 

            img_url = self._get_img_url(event)
            uid = event.unified_msg_origin

            # 1. 自动存图
            if img_url and not msg_str and not msg_str.startswith("/"):
                cooldown = self.local_config.get("auto_save_cooldown", 60)
                last_save = getattr(self, "last_auto_save_time", 0)
                if time.time() - last_save > cooldown:
                    # 异步执行，不阻塞主线程
                    asyncio.create_task(self.ai_evaluate_image(img_url, uid))

            # 2. 指令穿透
            if msg_str.startswith("/") or msg_str.startswith("！"):
                if uid in self.sessions:
                    if self.sessions[uid].get('timer_task'): self.sessions[uid]['timer_task'].cancel()
                    self.sessions[uid]['flush_event'].set()
                return

            # 3. 防抖逻辑
            debounce_time = self.local_config.get("debounce_time", 2.0)
            if debounce_time <= 0: return

            if uid in self.sessions:
                s = self.sessions[uid]
                if msg_str: s['buffer'].append(msg_str)
                if img_url: s['images'].append(img_url)
                if s.get('timer_task'): s['timer_task'].cancel()
                s['timer_task'] = asyncio.create_task(self._timer_coroutine(uid, debounce_time))
                event.stop_event()
                return

            flush_event = asyncio.Event()
            timer_task = asyncio.create_task(self._timer_coroutine(uid, debounce_time))
            self.sessions[uid] = {
                'buffer': [msg_str] if msg_str else [],
                'images': [img_url] if img_url else [],
                'flush_event': flush_event,
                'timer_task': timer_task
            }
            print(f"[Meme] 收到消息，开始防抖等待...")
            await flush_event.wait()

            if uid not in self.sessions: return
            s = self.sessions.pop(uid)
            merged_text = "\n".join(s['buffer']).strip()
            
            # 4. 注入小抄 (修复格式，放在末尾更隐蔽)
            if random.randint(1, 100) <= self.local_config.get("reply_prob", 50):
                all_tags = [i.get("tags") for i in self.data.values()]
                if all_tags:
                    # 随机选 20 个 tag
                    hint_tags = "、".join(random.sample(all_tags, min(20, len(all_tags))))
                    # 使用更清晰的 System Prompt 格式，避免混淆
                    prompt_inject = f"\n\n(System Hint: You have access to these memes: [{hint_tags}]. To send one, output exactly: MEME_TAG:tag_name inside your response)"
                    merged_text += prompt_inject

            event.message_str = merged_text
            event.message_obj.message = [Plain(merged_text)]
            print(f"[Meme] 防抖结束，放行: {merged_text[:30]}...")

        except Exception as e:
            print(f"ERROR inside handler: {e}")
            return

    # ==========================
    # 核心 2: 输出端分段 (增强正则与容错)
    # ==========================
    @filter.on_decorating_result(priority=0)
    async def on_decorate(self, event: AstrMessageEvent):
        if getattr(event, "__processed", False): return
        
        result = event.get_result()
        if not result: return
        
        text = ""
        if isinstance(result, list):
            for c in result:
                if isinstance(c, Plain): text += c.text
        elif hasattr(result, "chain"):
            for c in result.chain:
                if isinstance(c, Plain): text += c.text
        else: text = str(result)
            
        if not text: return
        setattr(event, "__processed", True)
        
        try:
            # 1. 增强版正则：兼容 [MEME_TAG:xxx] 和 MEME_TAG:xxx
            # 解释：找到 MEME_TAG: 后面直到 换行、空格、]、) 结束的字符
            mixed_chain = []
            parts = re.split(r"(\[?MEME_TAG:[^ \n\]\)]+\]?)", text) 
            
            has_tag = False
            for part in parts:
                clean_part = part.strip()
                if "MEME_TAG:" in clean_part:
                    has_tag = True
                    # 清理 tag 中的多余符号
                    tag = clean_part.replace("MEME_TAG:", "").replace("[", "").replace("]", "").replace("(", "").replace(")", "").strip()
                    path = self.find_best_match(tag)
                    if path: 
                        print(f"🎯 命中图片: {tag}")
                        mixed_chain.append(Image.fromFileSystem(path))
                    else:
                        mixed_chain.append(Plain(f"[缺: {tag}]"))
                elif clean_part:
                    # 只有纯文本才加进去
                    mixed_chain.append(Plain(part)) # 这里保留原始 part 以维持空格格式
            
            if not has_tag and len(text) < 50: return

            # 2. 智能分段
            segments = self.smart_split(mixed_chain)
            
            # 3. 拟人发送
            delay_base = self.local_config.get("delay_base", 0.5)
            delay_factor = self.local_config.get("delay_factor", 0.1)
            
            for i, seg in enumerate(segments):
                txt_len = sum(len(c.text) for c in seg if isinstance(c, Plain))
                wait = delay_base + (txt_len * delay_factor)
                
                mc = MessageChain()
                mc.chain = seg
                await self.context.send_message(event.unified_msg_origin, mc)
                
                if i < len(segments) - 1: await asyncio.sleep(wait)
            
            # 4. 完美终止原消息
            event.set_result(None)

        except Exception as e:
            print(f"分段发送出错: {e}")

    # ==========================
    # 核心 3: 图片处理 (压缩 + 自动保存)
    # ==========================
    def compress_image(self, file_path, quality=75):
        """压缩图片并转换为JPG"""
        try:
            with PILImage.open(file_path) as img:
                # 转换为 RGB 模式（防止 RGBA 存 JPG 报错）
                if img.mode in ("RGBA", "P"): img = img.convert("RGB")
                # 限制最大尺寸，例如宽度不超过 1024
                if img.width > 1024:
                    ratio = 1024 / img.width
                    new_height = int(img.height * ratio)
                    img = img.resize((1024, new_height), PILImage.LANCZOS)
                
                # 覆盖保存
                img.save(file_path, "JPEG", quality=quality)
                return True
        except Exception as e:
            print(f"图片压缩失败: {e}")
            return False

    async def ai_evaluate_image(self, img_url, uid):
        try:
            self.last_auto_save_time = time.time()
            provider = self.context.get_using_provider()
            if not provider: return
            
            # 提示词保持不变...
            prompt = """你正在帮我整理一个 QQ 表情包素材库...（同原代码）..."""
            
            resp = await provider.text_chat(prompt, session_id=None, image_urls=[img_url])
            content = (getattr(resp, "completion_text", None) or getattr(resp, "text", "")).strip()
            
            if "YES" in content.upper():
                tag = content.split('\n')[-1].replace("标签", "").strip() or "未分类"
                print(f"🖤 [自动进货] {tag}")
                
                # 保存图片
                saved_path = await self._save_img(img_url, tag, "auto")
                
                # 发送反馈给用户！
                if saved_path:
                    chain = MessageChain().message([Plain(f"已收录表情包：{tag}")])
                    await self.context.send_message(uid, chain)
                    
        except Exception as e:
            print(f"AI 审图出错: {e}")

    # ==========================
    # Web 后台 (含进度条支持)
    # ==========================
    async def start_web_server(self):
        app = web.Application()
        app._client_max_size = 50 * 1024 * 1024 
        
        app.router.add_get("/", self.h_idx)
        app.router.add_post("/upload", self.h_up)
        app.router.add_post("/batch_delete", self.h_del)
        app.router.add_post("/update_tag", self.h_tag)
        app.router.add_get("/get_config", self.h_gcf)
        app.router.add_post("/update_config", self.h_ucf)
        app.router.add_get("/backup", self.h_backup)
        app.router.add_post("/restore", self.h_restore)
        app.router.add_static("/images/", path=self.img_dir)
        
        runner = web.AppRunner(app); await runner.setup()
        port = self.local_config.get("web_port", 5000)
        site = web.TCPSite(runner, "0.0.0.0", port)
        await site.start()
        print(f"WebUI: http://localhost:{port}")

    # --- Handlers ---
    async def h_idx(self,r): 
        # 确保读取最新数据
        return web.Response(text=self.read_file("index.html").replace("{{MEME_DATA}}", json.dumps(self.data)), content_type="text/html")
    
    async def h_up(self,r):
        rd = await r.multipart(); tag="未分类"
        while True:
            p = await rd.next()
            if not p: break
            if p.name == "file":
                fn = f"{int(time.time()*1000)}_{random.randint(100,999)}.jpg"
                fp = os.path.join(self.img_dir, fn)
                with open(fp, "wb") as f: f.write(await p.read())
                
                # 立即压缩
                self.compress_image(fp)
                
                self.data[fn] = {"tags": tag, "source": "manual"}
            elif p.name == "tags": tag = await p.text()
        self.save_data(); return web.Response(text="ok")
        
    async def h_del(self,r):
        for f in (await r.json()).get("filenames",[]):
            try: os.remove(os.path.join(self.img_dir, f)); del self.data[f]
            except: pass
        self.save_data(); return web.Response(text="ok")
        
    async def h_tag(self,r): d=await r.json(); self.data[d['filename']]['tags']=d['tags']; self.save_data(); return web.Response(text="ok")
    async def h_gcf(self,r): return web.json_response(self.local_config)
    
    async def h_ucf(self,r): 
        new_conf = await r.json()
        self.local_config.update(new_conf)
        self.save_config() # 确保写入文件
        return web.Response(text="ok")

    async def h_backup(self, r):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as z:
            for root, _, files in os.walk(self.img_dir):
                for file in files: z.write(os.path.join(root, file), f"images/{file}")
            if os.path.exists(self.data_file): z.write(self.data_file, "memes.json")
            if os.path.exists(self.config_file): z.write(self.config_file, "config.json")
        buffer.seek(0)
        return web.Response(body=buffer, headers={
            'Content-Disposition': f'attachment; filename="meme_backup_{int(time.time())}.zip"',
            'Content-Type': 'application/zip'
        })

    async def h_restore(self, r):
        reader = await r.multipart()
        field = await reader.next()
        if not field or field.name != 'file': return web.Response(status=400, text="No file")
        buffer = io.BytesIO(await field.read())
        try:
            with zipfile.ZipFile(buffer, 'r') as z:
                z.extractall(self.base_dir)
            self.data = self.load_data()
            self.local_config = self.load_config()
            return web.Response(text="ok")
        except Exception as e:
            return web.Response(status=500, text=str(e))

    # --- Utils ---
    def smart_split(self, chain):
        segs = []; buf = []
        for c in chain:
            if isinstance(c, Image):
                if buf: segs.append(buf[:]); buf = []
                segs.append([c]); continue
            if isinstance(c, Plain):
                txt = c.text; idx = 0; chunk = ""; stack = []
                while idx < len(txt):
                    char = txt[idx]
                    if char in self.pair_map: stack.append(char)
                    elif stack and char == self.pair_map[stack[-1]]: stack.pop()
                    
                    if not stack and char in "\n。？！?!":
                        chunk += char
                        if chunk.strip(): buf.append(Plain(chunk))
                        if buf: segs.append(buf[:]); buf = []
                        chunk = ""
                    else:
                        chunk += char
                    idx += 1
                if chunk: buf.append(Plain(chunk))
        if buf: segs.append(buf)
        return segs

    def find_best_match(self, query):
        best, score = None, 0
        for f, i in self.data.items():
            t = i.get("tags", "")
            if query in t: return os.path.join(self.img_dir, f)
            s = difflib.SequenceMatcher(None, query, t).ratio()
            if s > score: score = s; best = f
        if score > 0.4: return os.path.join(self.img_dir, best)
        return None
    
    def read_file(self, n): 
        with open(os.path.join(self.base_dir, n), "r", encoding="utf-8") as f: return f.read()
        
    async def _save_img(self, url, tag, src):
        try:
            async with aiohttp.ClientSession() as s:
                async with s.get(url) as r:
                    fn = f"{int(time.time())}.jpg"
                    fp = os.path.join(self.img_dir, fn)
                    with open(fp, "wb") as f: f.write(await r.read())
                    self.compress_image(fp) # 保存时自动压缩
                    self.data[fn] = {"tags": tag, "source": src}
                    self.save_data()
                    return fp
        except Exception as e:
            print(f"Save Img Error: {e}")
            return None
            
    def _get_img_url(self, e):
        for c in e.message_obj.message:
            if isinstance(c, Image): return c.url
        return None
    def load_config(self): return {**{"web_port":5000,"debounce_time":2.0,"reply_prob":50}, **(json.load(open(self.config_file)) if os.path.exists(self.config_file) else {})}
    def save_config(self): json.dump(self.local_config, open(self.config_file,"w"), indent=2)
    def load_data(self): return json.load(open(self.data_file)) if os.path.exists(self.data_file) else {}
    def save_data(self): json.dump(self.data, open(self.data_file,"w"), ensure_ascii=False)
