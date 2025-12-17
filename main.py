import os
import json
import random
import asyncio
import time
import re
import aiohttp
import difflib
from aiohttp import web

from astrbot.api.star import Context, Star, register
from astrbot.api.event import filter, AstrMessageEvent, MessageChain
from astrbot.api.event.filter import EventMessageType
from astrbot.core.message.components import Image, Plain

print("DEBUG: MemeMaster 终极版 (GitHub) 已加载")

@register("vv_meme_master", "MemeMaster", "防抖+表情包+拟人分段", "1.0.0")
class MemeMaster(Star):
    def __init__(self, context: Context, config: dict = None):
        super().__init__(context)
        # 强制使用绝对路径，防止 Docker 迷路
        self.base_dir = os.path.abspath(os.path.dirname(__file__))
        self.img_dir = os.path.join(self.base_dir, "images")
        self.data_file = os.path.join(self.base_dir, "memes.json")
        self.config_file = os.path.join(self.base_dir, "config.json")
        
        if not os.path.exists(self.img_dir): os.makedirs(self.img_dir, exist_ok=True)
            
        self.local_config = self.load_config()
        self.data = self.load_data()
        
        # 防抖会话存储
        self.sessions = {}
        # 分段保护符
        self.pair_map = {'“': '”', '《': '》', '（': '）', '(': ')', '[': ']', '{': '}'}

        # 启动 Web 服务 (带异常捕获)
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.start_web_server())
        except Exception as e:
            print(f"ERROR: Web后台启动失败: {e}")

    # ==========================
    # 核心 1: 输入端防抖 (Input)
    # ==========================
    async def _timer_coroutine(self, uid: str, duration: float):
        try:
            await asyncio.sleep(duration)
            if uid in self.sessions: self.sessions[uid]['flush_event'].set()
        except asyncio.CancelledError: pass

    @filter.event_message_type(EventMessageType.PRIVATE_MESSAGE, priority=50)
    async def handle_private_msg(self, event: AstrMessageEvent):
        """全自动拦截：鉴图 + 防抖 + 注入"""
        try:
            msg_str = (event.message_str or "").strip()
            img_url = self._get_img_url(event)
            uid = event.unified_msg_origin

            # 1. 自动存图检测 (非指令、纯图片)
            if img_url and not msg_str and not msg_str.startswith("/"):
                cooldown = self.local_config.get("auto_save_cooldown", 60)
                last_save = getattr(self, "last_auto_save_time", 0)
                if time.time() - last_save > cooldown:
                    asyncio.create_task(self.ai_evaluate_image(img_url))

            # 2. 指令穿透 (不防抖)
            if msg_str.startswith("/") or msg_str.startswith("！"):
                if uid in self.sessions:
                    if self.sessions[uid].get('timer_task'): self.sessions[uid]['timer_task'].cancel()
                    self.sessions[uid]['flush_event'].set()
                return

            # 3. 防抖逻辑
            debounce_time = self.local_config.get("debounce_time", 2.0)
            if debounce_time <= 0: return # 没开防抖直接过

            # 场景 A: 续杯
            if uid in self.sessions:
                s = self.sessions[uid]
                if msg_str: s['buffer'].append(msg_str)
                if img_url: s['images'].append(img_url)
                if s.get('timer_task'): s['timer_task'].cancel()
                s['timer_task'] = asyncio.create_task(self._timer_coroutine(uid, debounce_time))
                event.stop_event() # 拦截！
                return

            # 场景 B: 新杯
            flush_event = asyncio.Event()
            timer_task = asyncio.create_task(self._timer_coroutine(uid, debounce_time))
            self.sessions[uid] = {
                'buffer': [msg_str] if msg_str else [],
                'images': [img_url] if img_url else [],
                'flush_event': flush_event,
                'timer_task': timer_task
            }
            print(f"[Meme] 收到消息，开始防抖等待...")
            await flush_event.wait() # 挂起等待

            # 场景 C: 结算
            if uid not in self.sessions: return
            s = self.sessions.pop(uid)
            merged_text = "\n".join(s['buffer']).strip()
            
            # 4. 注入小抄 (Cheat Sheet)
            if random.randint(1, 100) <= self.local_config.get("reply_prob", 50):
                all_tags = [i.get("tags") for i in self.data.values()]
                if all_tags:
                    # 随机选 20 个标签提示给 AI
                    hint_tags = "、".join(random.sample(all_tags, min(20, len(all_tags))))
                    merged_text += f"\n\n[System]\nAvailable Memes: {hint_tags}\nTo use, reply: MEME_TAG:tag_name"

            # 5. 放行
            event.message_str = merged_text
            event.message_obj.message = [Plain(merged_text)] # 简化处理，防抖后只传文本给 LLM
            # 图片暂不回传给 LLM，节省 token，反正 LLM 看不懂图的具体内容（除非 Vision 开启）
            print(f"[Meme] 防抖结束，放行: {merged_text[:20]}...")

        except Exception as e:
            print(f"ERROR inside handler: {e}")
            # 出错了就放行，别卡死
            return

    # ==========================
    # 核心 2: 输出端分段 (Output)
    # ==========================
    @filter.on_decorating_result(priority=0)
    async def on_decorate(self, event: AstrMessageEvent):
        """拦截 LLM 回复 -> 换图 -> 分段 -> 发送"""
        if getattr(event, "__processed", False): return
        
        result = event.get_result()
        if not result: return
        
        # 提取纯文本
        text = ""
        if isinstance(result, list):
            for c in result:
                if isinstance(c, Plain): text += c.text
        elif hasattr(result, "chain"):
            for c in result.chain:
                if isinstance(c, Plain): text += c.text
        else: text = str(result)
            
        if not text: return
        setattr(event, "__processed", True) # 标记已处理
        
        try:
            # 1. 解析标签 MEME_TAG:xxx
            mixed_chain = []
            parts = re.split(r"(MEME_TAG:[^\s\n]+)", text)
            for part in parts:
                if part.startswith("MEME_TAG:"):
                    tag = part.replace("MEME_TAG:", "").strip().replace("]", "").replace(")", "")
                    path = self.find_best_match(tag)
                    if path: 
                        print(f"🎯 命中图片: {tag}")
                        mixed_chain.append(Image.fromFileSystem(path))
                elif part:
                    mixed_chain.append(Plain(part))
            
            if not mixed_chain: return

            # 2. 智能分段
            segments = self.smart_split(mixed_chain)
            
            # 3. 拟人发送
            delay_base = self.local_config.get("delay_base", 0.5)
            delay_factor = self.local_config.get("delay_factor", 0.1)
            
            for i, seg in enumerate(segments):
                # 计算打字时间
                txt_len = sum(len(c.text) for c in seg if isinstance(c, Plain))
                wait = delay_base + (txt_len * delay_factor)
                
                # 发送
                mc = MessageChain(); mc.chain = seg
                await self.context.send_message(event.unified_msg_origin, mc)
                
                # 睡一会
                if i < len(segments) - 1: await asyncio.sleep(wait)
            
            # 4. 清空原始回复 (防止 AstrBot 重复发)
            event.set_result([])

        except Exception as e:
            print(f"分段发送出错: {e}")
            # 出错不处理，让 AstrBot 默认发

    # ==========================
    # 核心 3: 辅助与 Web
    # ==========================
    async def ai_evaluate_image(self, img_url):
        try:
            self.last_auto_save_time = time.time()
            provider = self.context.get_using_provider()
            if not provider: return
            prompt = """你正在帮我整理一个 QQ 表情包素材库。

请判断这张图片是否“值得被保存”，
作为未来聊天中可能会使用的表情包素材。
配文是：“{context_text}”。

判断时请注意：
- 这是一个偏二次元 / meme 使用环境
- 常见来源包括：chiikawa、这狗、线条小狗、多栋、猫meme 等
- 不要过度严肃，也不要把普通照片当成表情包

如果这张图不适合做表情包，请只回复：
NO

如果适合，请严格按下面格式回复（不要多余内容）：

YES
<名称>:<一句自然语言解释这个表情包在什么语境下使用>

规则：
1. 如果你能明确判断这是某个常见 IP、角色或 meme 系列，
   请直接使用大家普遍认得的名字作为「名称」
   例如：chiikawa、这狗、线条小狗、多栋、猫meme
2. 如果无法确定具体 IP，不要强行猜测，
   请使用一个简短的情绪或语气概括作为「名称」
3. 冒号后必须是一句完整、自然的“使用说明”，
   描述人在什么情况下会用这个表情包"""
            resp = await provider.text_chat(prompt, session_id=None, image_urls=[img_url])
            content = (getattr(resp, "completion_text", None) or getattr(resp, "text", "")).strip()
            if "YES" in content.upper():
                tag = content.split('\n')[-1].replace("标签", "").strip() or "未分类"
                print(f"🖤 [自动进货] {tag}")
                await self._save_img(img_url, tag, "auto")
        except: pass

    def smart_split(self, chain):
        # 简化版智能分段，保护括号
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
                    
                    # 只有不在括号里，且遇到换行或句号才分段
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
            if query in t: return os.path.join(self.img_dir, f) # 精确命中
            s = difflib.SequenceMatcher(None, query, t).ratio()
            if s > score: score = s; best = f
        if score > 0.4: return os.path.join(self.img_dir, best)
        return None

    # --- Web Handler 省略细节，功能与之前一致，加上了路径保护 ---
    async def start_web_server(self):
        app = web.Application()
        app.router.add_get("/", self.h_idx)
        app.router.add_post("/upload", self.h_up)
        app.router.add_post("/batch_delete", self.h_del)
        app.router.add_post("/update_tag", self.h_tag)
        app.router.add_get("/get_config", self.h_gcf)
        app.router.add_post("/update_config", self.h_ucf)
        app.router.add_static("/images/", path=self.img_dir)
        runner = web.AppRunner(app); await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", self.local_config.get("web_port", 5000))
        await site.start()
        print(f"WebUI: http://localhost:{self.local_config.get('web_port', 5000)}")

    async def h_idx(self,r): return web.Response(text=self.read_file("index.html").replace("{{MEME_DATA}}", json.dumps(self.data)), content_type="text/html")
    async def h_up(self,r):
        rd = await r.multipart(); tag="未分类"
        while True:
            p = await rd.next()
            if not p: break
            if p.name == "file":
                fn = f"{int(time.time()*1000)}.jpg" # 强制重命名，防中文乱码
                with open(os.path.join(self.img_dir, fn), "wb") as f: f.write(await p.read())
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
    async def h_ucf(self,r): self.local_config.update(await r.json()); self.save_config(); return web.Response(text="ok")

    # --- IO Utils ---
    def read_file(self, n): 
        with open(os.path.join(self.base_dir, n), "r", encoding="utf-8") as f: return f.read()
    async def _save_img(self, url, tag, src):
        async with aiohttp.ClientSession() as s:
            async with s.get(url) as r:
                fn = f"{int(time.time())}.jpg"
                with open(os.path.join(self.img_dir, fn), "wb") as f: f.write(await r.read())
                self.data[fn] = {"tags": tag, "source": src}; self.save_data()
    def _get_img_url(self, e):
        for c in e.message_obj.message:
            if isinstance(c, Image): return c.url
        return None
    def load_config(self): return {**{"web_port":5000,"debounce_time":2.0,"reply_prob":50}, **(json.load(open(self.config_file)) if os.path.exists(self.config_file) else {})}
    def save_config(self): json.dump(self.local_config, open(self.config_file,"w"), indent=2)
    def load_data(self): return json.load(open(self.data_file)) if os.path.exists(self.data_file) else {}
    def save_data(self): json.dump(self.data, open(self.data_file,"w"), ensure_ascii=False)
