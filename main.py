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
# 阴历支持
try:
    from lunar_python import Solar
    HAS_LUNAR = True
except ImportError:
    HAS_LUNAR = False

from astrbot.api.star import Context, Star, register
from astrbot.api.event import filter, AstrMessageEvent, MessageChain
from astrbot.api.event.filter import EventMessageType
from astrbot.core.message.components import Image, Plain

print("DEBUG: MemeMaster Pro (Final v11 - Fixed & Enhanced) Loaded")

@register("vv_meme_master", "Vvivloy", "防抖/图库/记忆/热重载", "5.1.0")
class MemeMaster(Star):
    def __init__(self, context: Context, config: dict = None):
        super().__init__(context)
        self.base_dir = os.path.abspath(os.path.dirname(__file__))
        self.img_dir = os.path.join(self.base_dir, "images")
        self.data_file = os.path.join(self.base_dir, "memes.json")
        self.config_file = os.path.join(self.base_dir, "config.json")
        self.memory_file = os.path.join(self.base_dir, "memory.txt") 
        self.buffer_file = os.path.join(self.base_dir, "buffer.json") 
        
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.running = True
        
        if not os.path.exists(self.img_dir): os.makedirs(self.img_dir, exist_ok=True)
            
        # 初始化配置（带热重载时间戳）
        self.config_mtime = 0
        self.local_config = self.load_config()
        if "web_token" not in self.local_config:
            self.local_config["web_token"] = "admin123"
            self.save_config()

        self.data = self.load_data()
        
        # 运行时状态
        self.chat_history_buffer = self.load_buffer_from_disk()
        self.current_summary = self.load_memory()
        self.msg_count = 0
        self.img_hashes = {} 
        self.sessions = {} 
        self.is_summarizing = False
        self.last_active_time = time.time()
        self.last_auto_save_time = 0 # 修复鉴图冷却变量
        
        self.pair_map = {'“': '”', '《': '》', '（': '）', '(': ')', '[': ']', '{': '}'}
        self.right_pairs = {v: k for k, v in self.pair_map.items()}

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.start_web_server())
            loop.create_task(self._init_image_hashes())
            loop.create_task(self._lonely_watcher()) 
            print("✅ [Meme] 核心服务启动成功！已启用配置热重载。", flush=True)
        except Exception as e:
            print(f"ERROR: 任务启动失败: {e}")

    def __del__(self):
        self.running = False 

    # ===============================================================
    # 辅助：智能配置加载 (修复“改了20秒无效”的问题)
    # ===============================================================
    def load_config(self):
        default = {
            "web_port": 5000, 
            "debounce_time": 3.0, 
            "reply_prob": 50, 
            "auto_save_cooldown": 60, 
            "memory_interval": 20, 
            "summary_threshold": 40, 
            "proactive_interval": 0,
            "delay_base": 0.5,
            "delay_factor": 0.1
        }
        
        if os.path.exists(self.config_file):
            try:
                # 记录文件修改时间
                mtime = os.path.getmtime(self.config_file)
                self.config_mtime = mtime
                
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    content = json.load(f)
                    default.update(content)
                # print(f"🔧 [Meme] 配置已加载 (防抖: {default.get('debounce_time')}s)")
            except json.JSONDecodeError as e:
                print(f"❌❌❌ [Meme] 配置文件格式错误！请检查逗号/引号！使用默认值。错误信息: {e}", flush=True)
            except Exception as e:
                print(f"⚠️ [Meme] 读取配置失败: {e}", flush=True)
        return default

    def check_config_reload(self):
        """检查配置文件是否在运行时被修改"""
        if os.path.exists(self.config_file):
            try:
                mtime = os.path.getmtime(self.config_file)
                if mtime > self.config_mtime:
                    print(f"🔄 [Meme] 检测到配置文件修改，正在热重载...", flush=True)
                    self.local_config = self.load_config()
                    print(f"✅ [Meme] 热重载完成！当前防抖时间: {self.local_config.get('debounce_time')}秒", flush=True)
            except: pass

    # ===============================================================
    # 核心 1：输入处理
    # ===============================================================
    async def _debounce_timer(self, uid: str, duration: float):
        try:
            await asyncio.sleep(duration)
            if uid in self.sessions: 
                self.sessions[uid]['flush_event'].set()
        except asyncio.CancelledError: 
            pass

    @filter.event_message_type(EventMessageType.PRIVATE_MESSAGE, priority=50)
    @filter.event_message_type(EventMessageType.GROUP_MESSAGE, priority=50)
    async def handle_input(self, event: AstrMessageEvent):
        try:
            # 0. 检查配置热更新
            self.check_config_reload()

            # 1. 基础检查
            user_id = str(event.message_obj.sender.user_id)
            if user_id == str(self.context.get_current_provider_bot().self_id): return
            
            # 获取消息内容
            msg_str = (event.message_str or "").strip()
            uid = event.unified_msg_origin
            img_urls = self._get_all_img_urls(event)

            # 打印日志 (修复日志丢失问题)
            if msg_str or img_urls:
                info = f"{msg_str[:15]}..." if msg_str else "[图片]"
                print(f"📨 [Meme] 收到消息: {info} (图:{len(img_urls)}) 来自: {user_id}", flush=True)

            self.last_active_time = time.time()
            self.last_uid = uid
            self.last_session_id = event.session_id

            # 2. 暗线任务：自动进货 (修复日志和冷却)
            if img_urls and not msg_str.startswith("/"):
                cd = float(self.local_config.get("auto_save_cooldown", 60))
                if time.time() - self.last_auto_save_time > cd:
                    print(f"🕵️ [Meme] 触发自动鉴图 ({len(img_urls)}张) - 冷却已就绪", flush=True)
                    self.last_auto_save_time = time.time() # 立即重置冷却
                    for url in img_urls:
                        asyncio.create_task(self.ai_evaluate_image(url, msg_str))
                else:
                    # print(f"⏳ [Meme] 自动鉴图冷却中...", flush=True)
                    pass

            # 3. 指令穿透
            if msg_str.startswith(("/", "！", "!")):
                print(f"⚡ [Meme] 指令穿透", flush=True)
                if uid in self.sessions:
                    if self.sessions[uid].get('timer_task'): 
                        self.sessions[uid]['timer_task'].cancel()
                    self.sessions[uid]['flush_event'].set()
                return 

            # 4. 核心防抖逻辑
            try: debounce_time = float(self.local_config.get("debounce_time", 3.0))
            except: debounce_time = 3.0

            if debounce_time <= 0:
                pass 
            else:
                # Case A: 追加模式
                if uid in self.sessions:
                    s = self.sessions[uid]
                    if msg_str: s['queue'].append({'type': 'text', 'content': msg_str})
                    for url in img_urls: s['queue'].append({'type': 'image', 'url': url})
                    
                    if s.get('timer_task'): s['timer_task'].cancel()
                    s['timer_task'] = asyncio.create_task(self._debounce_timer(uid, debounce_time))
                    
                    event.stop_event()
                    print(f"🔄 [Meme] 防抖追加 (队列: {len(s['queue'])} | 等待重置为 {debounce_time}s)", flush=True)
                    return 

                # Case B: 启动模式
                print(f"🆕 [Meme] 启动防抖计时 (等待 {debounce_time}s)...", flush=True)
                flush_event = asyncio.Event()
                timer_task = asyncio.create_task(self._debounce_timer(uid, debounce_time))
                
                initial_queue = []
                if msg_str: initial_queue.append({'type': 'text', 'content': msg_str})
                for url in img_urls: initial_queue.append({'type': 'image', 'url': url})

                self.sessions[uid] = {
                    'queue': initial_queue,
                    'flush_event': flush_event,
                    'timer_task': timer_task
                }
                
                await flush_event.wait()
                
                print(f"⏰ [Meme] 倒计时结束，开始打包发送给 LLM", flush=True)

                if uid not in self.sessions: return 
                s = self.sessions.pop(uid)
                queue = s['queue']
                
                if not queue: return

                combined_text_list = []
                combined_images = []
                for item in queue:
                    if item['type'] == 'text': combined_text_list.append(item['content'])
                    elif item['type'] == 'image': combined_images.append(item['url'])
                
                msg_str = " ".join(combined_text_list)
                img_urls = combined_images

            # 5. 上下文与记忆注入
            self.msg_count += 1
            
            # 记录 User 消息到 buffer (确保有日志)
            img_mark = f" [Image*{len(img_urls)}]" if img_urls else ""
            log_entry = f"User: {msg_str}{img_mark}"
            self.chat_history_buffer.append(log_entry)
            self.save_buffer_to_disk()
            
            # 构造 System Prompt
            time_info = self.get_full_time_str()
            system_context = [f"Time: {time_info}"]
            
            if self.current_summary:
                system_context.append(f"Long-term Memory: {self.current_summary}")

            try: reply_prob = int(self.local_config.get("reply_prob", 50))
            except: reply_prob = 50
            
            if random.randint(1, 100) <= reply_prob:
                all_tags = [v.get("tags", "").split(":")[0].strip() for v in self.data.values()]
                if all_tags:
                    hints = random.sample(all_tags, min(15, len(all_tags)))
                    hint_str = " ".join([f"<MEME:{h}>" for h in hints])
                    system_context.append(f"Meme Hints: {hint_str}")

            final_text = f"{msg_str}\n\n(System Context: {' | '.join(system_context)})"
            
            chain = [Plain(final_text)]
            for url in img_urls:
                chain.append(Image.fromURL(url))
            
            event.message_str = final_text
            event.message_obj.message = chain
            
            print(f"🚀 [Meme] 最终放行: {msg_str[:20]}... (SystemContext已注入)", flush=True)

        except Exception as e:
            import traceback
            print(f"❌ [Meme] 严重错误: {e}")
            traceback.print_exc()

    # ===============================================================
    # 核心 2：输出处理
    # ===============================================================
    @filter.on_decorating_result(priority=0)
    async def on_output(self, event: AstrMessageEvent):
        if getattr(event, "__meme_processed", False): return
        
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
        setattr(event, "__meme_processed", True)
        
        original_text = text
        text = self.clean_markdown(text)
        if text != original_text:
            print(f"🧹 [Meme] 检测到 Markdown 格式，已自动净化")
        
        print(f"📤 [Meme] 捕获 AI 回复内容: {text[:30]}...", flush=True)

        clean_text_for_log = re.sub(r"\(System Context:.*?\)", "", text).strip()
        self.chat_history_buffer.append(f"AI: {clean_text_for_log}")
        self.save_buffer_to_disk()
        
        if not self.is_summarizing:
            asyncio.create_task(self.check_and_summarize())

        try:
            pattern = r"(<MEME:.*?>|MEME_TAG:\s*[\S]+)"
            parts = re.split(pattern, text)
            mixed_chain = []
            has_meme = False
            
            for part in parts:
                tag = None
                if part.startswith("<MEME:"): tag = part[6:-1].strip()
                elif "MEME_TAG:" in part: tag = part.replace("MEME_TAG:", "").strip()
                
                if tag:
                    path = self.find_best_match(tag)
                    if path: 
                        print(f"🎯 [Meme] 命中表情包关键词: [{tag}] -> 准备发送图片", flush=True)
                        mixed_chain.append(Image.fromFileSystem(path))
                        has_meme = True
                    else:
                        print(f"⚠️ [Meme] 关键词 [{tag}] 未找到对应图片，忽略")
                elif part:
                    clean_part = part.replace("(System Context:", "").replace(")", "").strip()
                    if clean_part: mixed_chain.append(Plain(clean_part))
            
            if not has_meme and len(text) < 50 and "\n" not in text: return

            segments = self.smart_split(mixed_chain)
            delay_base = self.local_config.get("delay_base", 0.5)
            delay_factor = self.local_config.get("delay_factor", 0.1)
            
            print(f"🗣️ [Meme] 内容已分段，共 {len(segments)} 段，开始模拟打字发送...", flush=True)
            
            for i, seg in enumerate(segments):
                txt_len = sum(len(c.text) for c in seg if isinstance(c, Plain))
                wait = delay_base + (txt_len * delay_factor)
                
                mc = MessageChain()
                mc.chain = seg
                await self.context.send_message(event.unified_msg_origin, mc)
                if i < len(segments) - 1: await asyncio.sleep(wait)
            
            event.set_result(None) 

        except Exception as e:
            print(f"❌ [Meme] 输出处理出错: {e}")

    # ===============================================================
    # 功能逻辑
    # ===============================================================
    async def ai_evaluate_image(self, img_url, context_text=""):
        try:
            # print(f"🔍 [Meme] 正在请求图片数据...", flush=True)
            img_data = None
            async with aiohttp.ClientSession() as s:
                async with s.get(img_url) as r:
                    if r.status == 200: img_data = await r.read()
            if not img_data: return

            current_hash = await self._calc_hash_async(img_data)

            if current_hash:
                for _, exist_hash in self.img_hashes.items():
                    if bin(int(current_hash, 16) ^ int(exist_hash, 16)).count('1') <= 5:
                        print(f"♻️ [自动进货] 重复图片 (Hash碰撞)，跳过。", flush=True)
                        return

            provider = self.context.get_using_provider()
            if not provider: return
            
            default_prompt = "判断这张图是否适合做表情包。适合回YES并给出<名称>:说明，不适合回NO。"
            raw_prompt = self.local_config.get("ai_prompt", default_prompt)
            
            if "{context_text}" in raw_prompt:
                prompt = raw_prompt.replace("{context_text}", context_text)
            else:
                prompt = raw_prompt
            
            # print(f"📤 [自动进货] 发送 LLM 鉴定请求...", flush=True)
            resp = await provider.text_chat(prompt, session_id=None, image_urls=[img_url])
            content = (getattr(resp, "completion_text", None) or getattr(resp, "text", "")).strip()
            
            if "YES" in content:
                match = re.search(r"<(?P<tag>.*?)>[:：]?(?P<desc>.*)", content)
                if match:
                    full_tag = f"{match.group('tag').strip()}: {match.group('desc').strip()}"
                    print(f"🖤 [自动进货] 鉴定通过: {full_tag} -> 正在入库", flush=True)
                    
                    comp, ext = await self._compress_image(img_data)
                    fn = f"{int(time.time())}{ext}"
                    with open(os.path.join(self.img_dir, fn), "wb") as f: f.write(comp)
                    
                    self.data[fn] = {"tags": full_tag, "source": "auto", "hash": current_hash}
                    if current_hash: self.img_hashes[fn] = current_hash
                    self.save_data()
            else:
                pass
                # print(f"🗑️ [自动进货] 鉴定不通过: {content[:20]}", flush=True)
        except Exception as e:
            print(f"❌ [自动进货] 出错: {e}")

    async def _lonely_watcher(self):
        while self.running: 
            await asyncio.sleep(60) 
            # 同样每次循环检查配置
            self.check_config_reload()
            
            interval = self.local_config.get("proactive_interval", 0)
            if interval <= 0: continue
            
            q_start = self.local_config.get("quiet_start", -1)
            q_end = self.local_config.get("quiet_end", -1)
            if q_start != -1 and q_end != -1:
                h = datetime.datetime.now().hour
                is_quiet = False
                if q_start > q_end: 
                    if h >= q_start or h < q_end: is_quiet = True
                else:
                    if q_start <= h < q_end: is_quiet = True
                if is_quiet: continue

            if time.time() - self.last_active_time > (interval * 60):
                self.last_active_time = time.time() 
                provider = self.context.get_using_provider()
                uid = getattr(self, "last_uid", None)
                sid = getattr(self, "last_session_id", None)
                
                if provider and uid:
                    print(f"👋 [Meme] 主动发起聊天...", flush=True)
                    time_info = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
                    prompt = f"Time: {time_info}. User silent for {interval} mins. Memory: {self.current_summary}. Initiate conversation naturally."
                    
                    try:
                        resp = await provider.text_chat(prompt, session_id=sid)
                        text = (getattr(resp, "completion_text", None) or getattr(resp, "text", "")).strip()
                        if text:
                            self.chat_history_buffer.append(f"AI (Proactive): {text}")
                            self.save_buffer_to_disk()
                            mc = MessageChain([Plain(text)])
                            await self.context.send_message(uid, mc)
                    except: pass

    async def check_and_summarize(self):
        threshold = self.local_config.get("summary_threshold", 40)
        current_len = len(self.chat_history_buffer)
        
        # 补回详细日志
        # print(f"📊 [Meme] 当前记忆池: {current_len}/{threshold}", flush=True)

        if current_len < threshold: return
        
        print(f"⚠️ [Meme] 触发记忆总结 ({current_len}/{threshold})，正在压缩...", flush=True)
        
        self.is_summarizing = True 
        try:
            batch = list(self.chat_history_buffer)
            provider = self.context.get_using_provider()
            if not provider: return
            
            history_text = "\n".join(batch)
            now_str = self.get_full_time_str() 
            
            prompt = f"""当前时间：{now_str}
                这是一段过去的对话记录。请将其总结为一段简练的“长期记忆”或“日记”。
                重点记录：用户的喜好、发生的重要事件、双方约定的事情。
                忽略：无意义的寒暄、重复的表情包指令。
                字数限制：200字以内。
                对话内容：
                {history_text}"""
            resp = await provider.text_chat(prompt, session_id=None)
            summary = (getattr(resp, "completion_text", None) or getattr(resp, "text", "")).strip()
            
            if summary:
                def write():
                    with open(self.memory_file, "a", encoding="utf-8") as f: 
                        f.write(f"\n\n--- {now_str} ---\n{summary}")
                await asyncio.get_running_loop().run_in_executor(self.executor, write)
                
                self.current_summary = self.load_memory()
                self.chat_history_buffer = self.chat_history_buffer[len(batch):]
                self.save_buffer_to_disk()
                # 补回了详细的成功日志
                print(f"✅ [Meme] 总结完成！已归档 {len(batch)} 条对话，长期记忆更新。", flush=True)
        except Exception as e:
            print(f"❌ [Meme] 总结失败: {e}", flush=True)
        finally:
            self.is_summarizing = False

    # ==========================
    # 工具函数 & Web Server (保持不变但增强稳定性)
    # ==========================
    async def _init_image_hashes(self):
        if not os.path.exists(self.img_dir): return
        loop = asyncio.get_running_loop()
        count = 0
        for f in os.listdir(self.img_dir):
            if not f.lower().endswith(('.jpg', '.png', '.jpeg', '.gif', '.webp')): continue
            if f in self.data and 'hash' in self.data[f] and self.data[f]['hash']:
                self.img_hashes[f] = self.data[f]['hash']
                count += 1
                continue
            try:
                path = os.path.join(self.img_dir, f)
                with open(path, "rb") as fl: content = fl.read()
                h = await self._calc_hash_async(content)
                if h: 
                    self.img_hashes[f] = h
                    if f not in self.data: self.data[f] = {"tags": "未分类", "source": "unknown"}
                    self.data[f]['hash'] = h
                    count += 1
            except: pass
        self.save_data()
        print(f"✅ [Meme] 图库索引加载完毕！有效指纹: {count}", flush=True)

    async def _calc_hash_async(self, image_data):
        def _sync():
            try:
                img = PILImage.open(io.BytesIO(image_data))
                if getattr(img, 'is_animated', False): img.seek(0)
                img = img.resize((9, 8), PILImage.Resampling.LANCZOS).convert('L')
                pixels = list(img.getdata())
                val = sum(2**i for i, v in enumerate([pixels[row*9+col] > pixels[row*9+col+1] for row in range(8) for col in range(8)]) if v)
                return hex(val)[2:]
            except: return None
        return await asyncio.get_running_loop().run_in_executor(self.executor, _sync)

    async def _compress_image(self, image_data: bytes):
        def _sync():
            try:
                img = PILImage.open(io.BytesIO(image_data))
                if getattr(img, 'is_animated', False): return image_data, ".gif"
                max_w = 400
                if img.width > max_w:
                    ratio = max_w / img.width
                    img = img.resize((max_w, int(img.height * ratio)), PILImage.Resampling.LANCZOS)
                buf = io.BytesIO()
                if img.mode != "RGB": img = img.convert("RGB")
                img.save(buf, format="JPEG", quality=75)
                return buf.getvalue(), ".jpg"
            except: return image_data, ".jpg"
        return await asyncio.get_running_loop().run_in_executor(self.executor, _sync)

    def _get_all_img_urls(self, e):
        urls = []
        if not e.message_obj or not e.message_obj.message: return urls
        for c in e.message_obj.message:
            if isinstance(c, Image): urls.append(c.url)
        return urls
    
    def _get_img_url(self, e): return (self._get_all_img_urls(e) or [None])[0]

    def smart_split(self, chain):
        segs = []; buf = []
        def flush(): 
            if buf: segs.append(buf[:]); buf.clear()
        for c in chain:
            if isinstance(c, Image): flush(); segs.append([c]); continue
            if isinstance(c, Plain):
                txt = c.text; idx = 0; chunk = ""; stack = []
                while idx < len(txt):
                    char = txt[idx]
                    if char in self.pair_map: stack.append(char)
                    elif stack and char == self.pair_map[stack[-1]]: stack.pop()
                    if not stack and char in "\n。？！?!":
                        chunk += char
                        while idx + 1 < len(txt) and txt[idx+1] in "\n。？！?!": idx += 1; chunk += txt[idx]
                        if chunk.strip(): buf.append(Plain(chunk))
                        flush(); chunk = ""
                    else: chunk += char
                    idx += 1
                if chunk: buf.append(Plain(chunk))
        flush(); return segs

    def find_best_match(self, query):
        best, score = None, 0
        for f, i in self.data.items():
            t = i.get("tags", "")
            if query in t: return os.path.join(self.img_dir, f)
            s = difflib.SequenceMatcher(None, query, t.split(":")[0]).ratio()
            if s > score: score = s; best = f
        if score > 0.4: return os.path.join(self.img_dir, best)
        return None
    
    def save_config(self): 
        try: json.dump(self.local_config, open(self.config_file,"w"), indent=2)
        except: pass
    def load_data(self): return json.load(open(self.data_file)) if os.path.exists(self.data_file) else {}
    def save_data(self): json.dump(self.data, open(self.data_file,"w"), ensure_ascii=False)
    def load_buffer_from_disk(self):
        try: return json.load(open(self.buffer_file, "r"))
        except: return []
    def save_buffer_to_disk(self):
        try: json.dump(self.chat_history_buffer, open(self.buffer_file, "w"), ensure_ascii=False)
        except: pass
    def load_memory(self):
        try: return open(self.memory_file, "r", encoding="utf-8").read()
        except: return ""
    def read_file(self, n): 
        try: return open(os.path.join(self.base_dir, n), "r", encoding="utf-8").read()
        except: return ""
    def check_auth(self, r): return r.query.get("token") == self.local_config.get("web_token")

    def get_full_time_str(self):
        now = datetime.datetime.now()
        time_str = now.strftime('%Y-%m-%d %H:%M')
        if HAS_LUNAR:
            try:
                lunar = Solar.fromYmdHms(now.year, now.month, now.day, now.hour, now.minute, now.second).getLunar()
                time_str += f" (农历{lunar.getMonthInChinese()}月{lunar.getDayInChinese()})"
            except: pass
        return time_str

    def clean_markdown(self, text):
        text = text.replace("**", "")
        text = text.replace("### ", "").replace("## ", "")
        if text.startswith("> "): text = text[2:]
        return text.strip()

    # ==========================
    # Web Server
    # ==========================
    async def start_web_server(self):
        app = web.Application()
        app._client_max_size = 100 * 1024 * 1024 
        app.router.add_get("/", self.h_idx)
        app.router.add_post("/upload", self.h_up)
        app.router.add_post("/batch_delete", self.h_del)
        app.router.add_post("/update_tag", self.h_tag)
        app.router.add_get("/get_config", self.h_gcf)
        app.router.add_post("/update_config", self.h_ucf)
        app.router.add_get("/backup", self.h_backup)
        app.router.add_post("/restore", self.h_restore)
        app.router.add_post("/slim_images", self.h_slim)
        app.router.add_static("/images/", path=self.img_dir)
        runner = web.AppRunner(app)
        await runner.setup()
        port = self.local_config.get("web_port", 5000)
        site = web.TCPSite(runner, "0.0.0.0", port)
        await site.start()
        print(f"🌐 [Meme] WebUI 管理后台已启动: http://localhost:{port}", flush=True)

    async def h_idx(self,r): 
        if not self.check_auth(r): return web.Response(status=403, text="Need ?token=xxx")
        token = self.local_config["web_token"]
        html = self.read_file("index.html").replace("{{MEME_DATA}}", json.dumps(self.data)).replace("admin123", token)
        return web.Response(text=html, content_type="text/html")
    async def h_up(self, r):
        if not self.check_auth(r): return web.Response(status=403)
        rd = await r.multipart(); tag="未分类"
        while True:
            p = await rd.next()
            if not p: break
            if p.name == "tags": tag = await p.text()
            elif p.name == "file":
                raw = await p.read()
                comp, ext = await self._compress_image(raw)
                fn = f"{int(time.time()*1000)}_{random.randint(100,999)}{ext}"
                with open(os.path.join(self.img_dir, fn), "wb") as f: f.write(comp)
                h = await self._calc_hash_async(comp) 
                self.data[fn] = {"tags": tag, "source": "manual", "hash": h}
                if h: self.img_hashes[fn] = h
        self.save_data(); return web.Response(text="ok")
    async def h_del(self,r):
        if not self.check_auth(r): return web.Response(status=403)
        for f in (await r.json()).get("filenames",[]):
            try: os.remove(os.path.join(self.img_dir,f)); del self.data[f]; self.img_hashes.pop(f, None)
            except: pass
        self.save_data(); return web.Response(text="ok")
    async def h_tag(self,r):
        if not self.check_auth(r): return web.Response(status=403)
        d=await r.json(); self.data[d['filename']]['tags']=d['tags']; self.save_data(); return web.Response(text="ok")
    async def h_gcf(self,r): return web.json_response(self.local_config)
    async def h_ucf(self,r):
        if not self.check_auth(r): return web.Response(status=403)
        self.local_config.update(await r.json()); self.save_config(); return web.Response(text="ok")
    
    async def h_backup(self,r):
        if not self.check_auth(r): return web.Response(status=403)
        b=io.BytesIO()
        with zipfile.ZipFile(b,'w',zipfile.ZIP_DEFLATED) as z:
            for root,_,files in os.walk(self.img_dir): 
                for f in files: z.write(os.path.join(root,f),f"images/{f}")
            if os.path.exists(self.data_file): z.write(self.data_file,"memes.json")
            if os.path.exists(self.config_file): z.write(self.config_file,"config.json")
            if os.path.exists(self.memory_file): z.write(self.memory_file, "memory.txt")
            if os.path.exists(self.buffer_file): z.write(self.buffer_file, "buffer.json")
        b.seek(0)
        return web.Response(body=b, headers={'Content-Disposition':'attachment; filename="meme_backup.zip"'})
    
    async def h_restore(self, r):
        if not self.check_auth(r): return web.Response(status=403, text="Forbidden")
        try:
            reader = await r.multipart()
            field = await reader.next()
            if not field or field.name != 'file': return web.Response(status=400, text="Invalid file")
            file_data = await field.read()
            if not file_data: return web.Response(status=400, text="Empty file")
            def unzip_action():
                with zipfile.ZipFile(io.BytesIO(file_data), 'r') as z: z.extractall(self.base_dir)
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(self.executor, unzip_action)
            self.data = self.load_data()
            self.local_config = self.load_config()
            self.current_summary = self.load_memory() 
            self.chat_history_buffer = self.load_buffer_from_disk()
            asyncio.create_task(self._init_image_hashes())
            return web.Response(text="ok")
        except Exception as e:
            return web.Response(status=500, text=f"Error: {str(e)}")

    async def h_slim(self, r):
        if not self.check_auth(r): return web.Response(status=403)
        loop = asyncio.get_running_loop()
        count = 0
        for f in os.listdir(self.img_dir):
            try:
                p = os.path.join(self.img_dir, f)
                with open(p, 'rb') as fl: raw = fl.read()
                nd, _ = await self._compress_image(raw)
                if len(nd) < len(raw):
                    with open(p, 'wb') as fl: fl.write(nd)
                    count += 1
            except: pass
        return web.Response(text=f"优化了 {count} 张")
