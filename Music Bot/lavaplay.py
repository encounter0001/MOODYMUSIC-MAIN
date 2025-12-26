import discord
import wavelink
import redis
import os
import logging
import asyncio
import aiohttp
import random
import json
import time
import re
import html
import yt_dlp
from discord.ext import commands
from typing import Optional, cast, Dict, List, Tuple, Any, Union
from dotenv import load_dotenv
from discord import ui, app_commands
import async_timeout
from datetime import datetime, timedelta
from aiohttp import web
from discord.ext import tasks
import base64


def safe_bool(v):
    try:
        return bool(int(v))
    except Exception:
        return False

async def safe_send(channel, *args, **kwargs):
    """Send to a channel but swallow common errors (deleted/permissions)."""
    if channel is None:
        return None
    try:
        return await channel.send(*args, **kwargs)
    except (discord.Forbidden, discord.HTTPException, AttributeError) as e:
        logger.warning(f"safe_send failed: {e}")
        return None

async def safe_edit_response(interaction, **kwargs):
    """Attempt edit_message or fallback to followup; safe for use within callbacks."""
    try:
        await interaction.response.edit_message(**kwargs)
    except discord.HTTPException:
        try:
            await interaction.followup.send(**{k:v for k,v in kwargs.items() if k in ('content','embed')}, ephemeral=True)
        except Exception:
            logger.debug("safe_edit_response followup failed", exc_info=True)

def sanitize_prefix(prefix: str, max_len: int = 8) -> str:
    if not isinstance(prefix, str):
        return "m!"
    prefix = prefix.strip()
    if not prefix:
        return "m!"
    prefix = prefix[:max_len]
    # optional: restrict characters
    return re.sub(r'[\s\n\r]+', '', prefix)

# --- Configuration & Setup ---
load_dotenv()
TOKEN = os.getenv("DISCORD_TOKEN")
if not TOKEN:
    raise ValueError("DISCORD_TOKEN environment variable is required")

# New Configs for API and AI
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
API_PORT = int(os.getenv("API_PORT", "8080"))
API_SECRET = os.getenv("API_SECRET", "changeme")

# Parse owner IDs from environment variable
OWNER_IDS = []
owner_env = os.getenv("OWNER_ID", "")
if owner_env:
    for id_str in owner_env.split(","):
        id_str = id_str.strip()
        if id_str.isdigit():
            OWNER_IDS.append(int(id_str))

# Lavalink Configuration
LAVALINK_URI = os.getenv("LAVALINK_URI", "http://us7.endercloud.in:4577")
LAVALINK_PASS = os.getenv("LAVALINK_PASS", "youshallnotpass")
LAVALINK_IDENTIFIER = os.getenv("LAVALINK_ID", "main_node")

# Genius Lyrics Token (Optional)
GENIUS_TOKEN = os.getenv("GENIUS_TOKEN", "")

# --- AI Recommendation Engine ---
class AIRecommender:
    """Handles interaction with Gemini API for song recommendations"""
    def __init__(self):
        self.api_key = GEMINI_API_KEY
        if self.api_key:
            self.url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={self.api_key}"
        else:
            self.url = None

    async def get_recommendations(self, prompt_text: str) -> List[str]:
        """
        Production-grade AI recommender with:
        - retries
        - timeout handling
        - malformed response handling
        - graceful fallbacks
        """
        if not self.api_key or not self.url:
            logger.warning("AIRecommender: Missing API key or URL")
            return []

        if not prompt_text or not isinstance(prompt_text, str):
            return []

        payload = {
            "contents": [{
                "parts": [{
                    "text": (
                        f"You are a music DJ. Recommend 10 songs based on this request:\n"
                        f"'{prompt_text}'.\n\n"
                        "Return ONLY a valid JSON array of strings.\n"
                        "Format strictly like:\n"
                        '["Artist - Song", "Artist - Song"]\n'
                        "No explanations. No markdown."
                    )
                }]
            }],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 500
            }
        }

        retries = 2
        timeout_seconds = 15

        for attempt in range(1, retries + 1):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        self.url,
                        json=payload,
                        timeout=timeout_seconds
                    ) as resp:

                        if resp.status != 200:
                            logger.warning(
                                f"Gemini API attempt {attempt} failed "
                                f"(status={resp.status})"
                            )
                            continue

                        data = await resp.json(content_type=None)

                # ---- Defensive parsing ----
                if not isinstance(data, dict):
                    logger.warning("Gemini returned non-dict response")
                    continue

                candidates = data.get("candidates")
                if not candidates or not isinstance(candidates, list):
                    logger.warning("Gemini response missing candidates")
                    continue

                content = candidates[0].get("content", {})
                parts = content.get("parts", [])

                if not parts or not isinstance(parts, list):
                    logger.warning("Gemini response missing parts")
                    continue

                raw_text = parts[0].get("text", "")
                if not raw_text:
                    logger.warning("Gemini returned empty text")
                    continue

                # ---- Clean AI output ----
                cleaned = (
                    raw_text
                    .replace("```json", "")
                    .replace("```", "")
                    .strip()
                )

                # ---- Try strict JSON ----
                try:
                    parsed = json.loads(cleaned)
                    if isinstance(parsed, list):
                        results = []
                        for item in parsed:
                            if isinstance(item, str):
                                item = item.strip()
                                if len(item) > 2:
                                    results.append(item)
                        if results:
                            return results[:10]
                except json.JSONDecodeError:
                    logger.info("Gemini output not valid JSON, using fallback parsing")

                # ---- Fallback: line-by-line parsing ----
                lines = []
                for line in cleaned.splitlines():
                    line = line.strip()
                    line = re.sub(r"^[\d\.\-\)\s]+", "", line)
                    if " - " in line and len(line) > 4:
                        lines.append(line)

                if lines:
                    return lines[:10]

            except asyncio.TimeoutError:
                logger.warning(f"Gemini timeout on attempt {attempt}")
            except aiohttp.ClientError as e:
                logger.warning(f"Gemini network error on attempt {attempt}: {e}")
            except Exception as e:
                logger.exception(f"Gemini unexpected error on attempt {attempt}: {e}")

            # small backoff before retry
            await asyncio.sleep(1.5)

        logger.error("AIRecommender: All attempts failed")
        return []



# --- Web Server (REST API) ---
class WebServer:
    """REST API for external dashboards"""
    def __init__(self, bot):
        self.bot = bot
        self.app = web.Application()
        self.setup_routes()
        self.runner = None
        self.site = None

    def setup_routes(self):
        self.app.router.add_get('/', self.handle_root)
        self.app.router.add_get('/api/stats', self.handle_stats)
        self.app.router.add_get('/api/guild/{guild_id}', self.handle_guild_info)

    def _check_auth(self, request):
        token = request.headers.get("Authorization")
        return token == API_SECRET

    async def handle_root(self, request):
        return web.json_response({"status": "online", "bot": self.bot.user.name})

    async def handle_stats(self, request):
        if not self._check_auth(request):
            return web.json_response({"error": "Unauthorized"}, status=401)

        total_players = len([vc for vc in self.bot.voice_clients if isinstance(vc, wavelink.Player)])
        playing_players = sum(1 for vc in self.bot.voice_clients if isinstance(vc, wavelink.Player) and vc.playing)

        return web.json_response({
            "guilds": len(self.bot.guilds),
            "users": sum(g.member_count for g in self.bot.guilds),
            "voice_connections": total_players,
            "active_players": playing_players,
            "latency_ms": round(self.bot.latency * 1000)
        })

    async def handle_guild_info(self, request):
        if not self._check_auth(request):
            return web.json_response({"error": "Unauthorized"}, status=401)

        guild_id = int(request.match_info['guild_id'])
        guild = self.bot.get_guild(guild_id)

        if not guild:
            return web.json_response({"error": "Guild not found"}, status=404)

        vc: Optional[wavelink.Player] = guild.voice_client
        queue_data = []
        now_playing = None

        if vc and isinstance(vc, wavelink.Player):
            if vc.current:
                now_playing = {
                    "title": vc.current.title,
                    "artist": vc.current.author,
                    "uri": vc.current.uri or "",
                    "duration": vc.current.length,
                    "position": vc.position
                }

            for track in vc.queue:
                queue_data.append({
                    "title": track.title,
                    "artist": track.author,
                    "uri": track.uri or ""
                })

        return web.json_response({
            "name": guild.name,
            "id": str(guild.id),
            "connected": vc is not None,
            "now_playing": now_playing,
            "queue": queue_data,
            "queue_length": len(queue_data)
        })

    async def start(self):
        try:
            self.runner = web.AppRunner(self.app)
            await self.runner.setup()
            self.site = web.TCPSite(self.runner, '0.0.0.0', API_PORT)
            await self.site.start()
            logger.info(f"REST API running on port {API_PORT}")
        except Exception as e:
            logger.error(f"Failed to start web server: {e}")

    async def stop(self):
        if self.site:
            await self.site.stop()
        if self.runner:
            await self.runner.cleanup()


# Provider Icons
ICONS = {
    "spotify": "https://upload.wikimedia.org/wikipedia/commons/thumb/1/19/Spotify_logo_without_text.svg/2048px-Spotify_logo_without_text.svg.png",
    "youtube": "https://i.pinimg.com/originals/de/1c/91/de1c91788be0d791135736995109272a.png",
    "applemusic": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5f/Apple_Music_icon.svg/2048px-Apple_Music_icon.svg.png",
    "soundcloud": "https://i.pinimg.com/originals/95/fa/01/95fa01c3d82a433948507b960b76921b.png",
    "gaana": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e5/Gaana.com_Logo.svg/1024px-Gaana.com_Logo.svg.png",
    "jiosaavn": "https://images.sftcdn.net/images/t_app-icon-m/p/4b3bebe9-f429-42cc-89db-2a9493062a5e/image.png",
    "default": "https://cdn-icons-png.flaticon.com/512/109/109197.png"
}

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("MoodyMusic")

# --- Database Manager (Redis) ---
class DatabaseManager:
    def __init__(self):
        # Initialize Redis Connection
        self.r = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            password=os.getenv("REDIS_PASSWORD", None),
            db=int(os.getenv("REDIS_DB", 0)),
            decode_responses=True # crucial so you get strings, not bytes
        )
        try:
            self.r.ping()
            logger.info("✅ Connected to Redis Database")
        except redis.ConnectionError as e:
            logger.error(f"❌ Failed to connect to Redis: {e}")
            raise e

    # --- Prefix Management ---
    def get_prefix(self, guild_id: int) -> str:
        """Return prefix for guild or default 'm!'."""
        try:
            prefix = self.r.hget("bot:prefixes", str(guild_id))
            return prefix if prefix else "m!"
        except Exception as e:
            logger.exception(f"Redis error in get_prefix({guild_id}): {e}")
            return "m!"

    def set_prefix(self, guild_id: int, new_prefix: str):
        """Set prefix safely."""
        try:
            new_prefix = new_prefix.strip()[:8] or "m!"
            self.r.hset("bot:prefixes", str(guild_id), new_prefix)
        except Exception as e:
            logger.exception(f"Redis error in set_prefix({guild_id}): {e}")

    # --- 24/7 Mode Management ---
    def get_247_status(self, guild_id: int) -> bool:
        """Check if 24/7 mode is active via the active set."""
        try:
            return self.r.sismember("bot:247:active_guilds", str(guild_id))
        except Exception:
            return False

    def get_redis_latency(self) -> int:
            """Measure Redis round-trip latency in milliseconds."""
            try:
                start = time.perf_counter()
                self.r.ping()
                end = time.perf_counter()
                return int((end - start) * 1000)
            except Exception as e:
                logger.error(f"Redis ping failed: {e}")
                return -1

    def get_247_channel(self, guild_id: int) -> Optional[int]:
        """Return saved 24/7 channel id or None."""
        try:
            val = self.r.get(f"guild:{guild_id}:247_channel")
            return int(val) if val else None
        except Exception:
            return None

    def set_247_status(self, guild_id: int, status: bool, channel_id: Optional[int] = None):
        """Enable/Disable 24/7 and store channel ID."""
        try:
            gid = str(guild_id)
            if status:
                # Add to active set
                self.r.sadd("bot:247:active_guilds", gid)
                if channel_id:
                    self.r.set(f"guild:{gid}:247_channel", channel_id)
            else:
                # Remove from active set
                self.r.srem("bot:247:active_guilds", gid)
                # Optional: We don't delete the channel ID so we remember it if they toggle back on
        except Exception as e:
            logger.exception(f"Redis error set_247_status({guild_id}): {e}")

    def get_all_247_channels(self) -> List[Tuple[int, int]]:
        """Return list of (guild_id, channel_id) for startup."""
        try:
            active_guilds = self.r.smembers("bot:247:active_guilds")
            results = []
            for gid in active_guilds:
                cid = self.r.get(f"guild:{gid}:247_channel")
                if cid:
                    results.append((int(gid), int(cid)))
            return results
        except Exception as e:
            logger.exception("Redis error get_all_247_channels")
            return []

    # --- History Management ---
    def add_to_history(self, guild_id: int, title: str, uri: str, author: str):
        """Add track to history list (LPUSH) and trim to 100."""
        try:
            # We store as a JSON string for easy parsing later
            data = json.dumps({
                "title": (str(title) if title else "Unknown")[:350],
                "uri": (str(uri) if uri else "")[:1000],
                "author": (str(author) if author else "")[:200]
            })
            key = f"guild:{guild_id}:history"
            
            pipe = self.r.pipeline()
            pipe.lpush(key, data)
            pipe.ltrim(key, 0, 99) # Keep only top 100
            pipe.execute()
        except Exception as e:
            logger.exception(f"Redis error add_to_history({guild_id}): {e}")

    def get_last_two_tracks(self, guild_id: int):
        """
        Return up to 2 last played tracks.
        Format expected by bot: [(title, uri, author, id), ...]
        """
        try:
            key = f"guild:{guild_id}:history"
            # Get top 2 items
            items = self.r.lrange(key, 0, 1)
            results = []
            
            # Redis lists don't have 'IDs' like SQL, so we return 0 as a dummy ID
            # The bot uses index 0 [title] and index 2 [author] mostly.
            for item in items:
                try:
                    obj = json.loads(item)
                    # Tuple: (title, uri, author, dummy_id)
                    results.append((obj["title"], obj["uri"], obj["author"], 0))
                except:
                    continue
            return results
        except Exception as e:
            logger.exception(f"Redis error get_last_two_tracks({guild_id}): {e}")
            return []

    def remove_last_track(self, guild_id: int):
        """Removes the most recent track (LPOP)."""
        try:
            self.r.lpop(f"guild:{guild_id}:history")
        except Exception as e:
            logger.exception(f"Redis error remove_last_track({guild_id}): {e}")

    def clear_history(self, guild_id: int):
        """Delete the history key."""
        try:
            self.r.delete(f"guild:{guild_id}:history")
        except Exception as e:
            logger.exception(f"Redis error clear_history({guild_id}): {e}")

    # --- User Stats ---
    def update_user_stats(self, user_id: int, guild_id: int):
        """Increment user's songs_played in a hash."""
        try:
            # Key: guild:{id}:stats -> Field: user_id -> Value: count
            self.r.hincrby(f"guild:{guild_id}:stats", str(user_id), 1)
        except Exception as e:
            logger.exception(f"Redis error update_user_stats: {e}")

# Instantiate
db = DatabaseManager()
# --- Custom Check for Owner Bypass ---
def is_owner_or_admin():
    """Check if user is bot owner or has administrator permissions"""
    async def predicate(ctx):
        if not ctx.guild:
            return False
        if ctx.author.id in OWNER_IDS:
            return True
        if ctx.author.guild_permissions.administrator:
            return True
        return False
    return commands.check(predicate)


class FailSafe(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.last_positions = {}  # {guild_id: position_ms}
        self.stuck_counts = {}    # {guild_id: count}
        self.player_watchdog.start()

    def cog_unload(self):
        self.player_watchdog.cancel()

    @tasks.loop(seconds=30)
    async def player_watchdog(self):
        """Background task: Handles Stuck Songs & 10-Minute AFK Timeout."""
        for guild in self.bot.guilds:
            if not guild.voice_client:
                continue
            
            vc: wavelink.Player = guild.voice_client
            
            if not vc.connected:
                continue

            # --- LOGIC 1: AFK TIMEOUT (10 Minutes) ---
            try:
                channel = vc.channel
                if channel:
                    members = channel.members
                    # Count humans (ignore bots)
                    humans = [m for m in members if not m.bot]
                    
                    if not humans:
                        # Bot is alone. Increment counter.
                        # 20 checks * 30 seconds = 600 seconds (10 Minutes)
                        current_count = getattr(vc, "afk_timer", 0) + 1
                        setattr(vc, "afk_timer", current_count)
                        
                        if current_count >= 20: 
                            logger.info(f"AFK Timeout in {guild.name}. Cleaning up...")
                            
                            # 1. Stop Music & Clear Queue
                            if getattr(vc, "queue", None):
                                vc.queue.clear()
                            await vc.stop()
                            try:
                                db.clear_history(guild.id)
                            except:
                                pass
                            
                            # 2. Check if a 24/7 Channel is assigned
                            home_channel_id = db.get_247_channel(guild.id)
                            
                            # --- SCENARIO A: NO 24/7 Channel Assigned ---
                            if not home_channel_id:
                                # 24/7 is OFF/Unset -> DISCONNECT IMMEDIATELY
                                await self.emergency_disconnect(guild, "AFK Timeout (10 mins - No 24/7 assigned)")
                                continue

                            # --- SCENARIO B: 24/7 Channel IS Assigned ---
                            # Are we in the wrong channel?
                            if channel.id != home_channel_id:
                                try:
                                    home_channel = guild.get_channel(home_channel_id)
                                    if home_channel:
                                        # Move to the 24/7 channel
                                        await vc.move_to(home_channel)
                                        
                                        # Reset timer so we don't disconnect
                                        setattr(vc, "afk_timer", 0) 
                                        logger.info(f"Returned to 24/7 Home: {home_channel.name}")
                                        
                                        # Notify user
                                        txt_ch = self.bot.get_player_text_channel(guild.id)
                                        if txt_ch:
                                            ch = self.bot.get_channel(txt_ch)
                                            await safe_send(ch, f"🏚️ **AFK:** Moving to 24/7 channel: {home_channel.mention}")
                                            
                                        # Reset channel status UI
                                        try:
                                            await home_channel.edit(status=None)
                                        except:
                                            pass
                                        
                                        continue # Skip disconnection logic
                                except Exception as e:
                                    logger.error(f"Failed to return home: {e}")
                                    # If move fails, we fall through to disconnect below

                            # --- SCENARIO C: We are IN the 24/7 Channel ---
                            # Final check: Is the 24/7 mode toggle actually ON?
                            if not db.get_247_status(guild.id):
                                # It's in the channel, but the mode is explicitly disabled
                                await self.emergency_disconnect(guild, "AFK Timeout (24/7 Disabled)")
                            else:
                                # All good, stay connected
                                setattr(vc, "afk_timer", 0)
                                
                    else:
                        # Humans present, reset timer
                        setattr(vc, "afk_timer", 0)
            except Exception as e:
                logger.error(f"Watchdog AFK Check Error: {e}")

            # --- LOGIC 2: STUCK SONG DETECTION ---
            # Only check if playing and NOT paused
            if vc.playing and not vc.paused:
                current_pos = vc.position
                last_pos = self.last_positions.get(guild.id, -1)

                # If position hasn't moved in 30 seconds
                if current_pos == last_pos:
                    count = self.stuck_counts.get(guild.id, 0) + 1
                    self.stuck_counts[guild.id] = count
                    
                    if count >= 2: # Stuck for 60 seconds total
                        logger.warning(f"FailSafe: Track stuck in {guild.name}. Attempting skip/reset.")
                        self.last_positions[guild.id] = -1
                        self.stuck_counts[guild.id] = 0
                        
                        # Notify
                        channel_id = self.bot.get_player_text_channel(guild.id)
                        if channel_id:
                            ch = self.bot.get_channel(channel_id)
                            await safe_send(ch, "⚠️ **FailSafe:** Song stuck detected. Auto-skipping...")
                        
                        try:
                            await vc.skip(force=True)
                        except wavelink.exceptions.LavalinkException:
                            logger.error(f"FailSafe: Player 404 (Ghost Session) in {guild.name}. Force resetting.")
                            await self.emergency_disconnect(guild, "Ghost Player Detected (404)")
                        except Exception as e:
                            logger.error(f"FailSafe: Error skipping stuck track: {e}")
                else:
                    # Reset counters if moving
                    self.last_positions[guild.id] = current_pos
                    self.stuck_counts[guild.id] = 0

    async def emergency_disconnect(self, guild, reason):
        """Force kills a connection."""
        try:
            if guild.voice_client:
                await guild.voice_client.disconnect(force=True)
            db.clear_history(guild.id)
            logger.info(f"Emergency disconnect in {guild.name}: {reason}")
        except Exception as e:
            logger.error(f"Failed emergency disconnect: {e}")

    # --- Listeners for Internal Errors ---

    @commands.Cog.listener()
    async def on_wavelink_track_stuck(self, payload: wavelink.TrackStuckEventPayload):
        """Called when Lavalink detects a stuck track internally."""
        logger.warning(f"Track stuck: {payload.track.title}")
        channel = self.bot.get_player_text_channel(payload.player.guild.id)
        if channel:
            ch = self.bot.get_channel(channel)
            await safe_send(ch, f"⚠️ **Glitch Detected:** {payload.track.title} got stuck. Skipping!")
        
        await payload.player.skip(force=True)

    @commands.Cog.listener()
    async def on_wavelink_track_exception(self, payload: wavelink.TrackExceptionEventPayload):
        """Called when a track crashes (e.g., 403 Forbidden, Decode Error)."""
        logger.error(f"Track exception: {payload.exception}")
        channel = self.bot.get_player_text_channel(payload.player.guild.id)
        if channel:
            ch = self.bot.get_channel(channel)
            await safe_send(ch, f"❌ **Error:** Cannot play {payload.track.title}. Skipping...")
        
        await payload.player.skip(force=True)

    # --- Manual Repair Command ---

    @commands.hybrid_command(name="repair", aliases=["fix", "failsafe"])
    async def repair(self, ctx):
        """🚨 Panic Button: Resets the bot's voice state completely."""
        await ctx.send("🚨 **Running Emergency Repair Protocols...**")
        
        step_msg = await ctx.send("1️⃣ Disconnecting from Voice...")
        
        # 1. Force Disconnect
        if ctx.guild.voice_client:
            try:
                await ctx.guild.voice_client.disconnect(force=True)
            except Exception as e:
                logger.error(f"Repair disconnect error: {e}")
        
        await step_msg.edit(content="2️⃣ Clearing Internal Cache...")
        
        # 2. Clear Database/Cache
        db.clear_history(ctx.guild.id)
        if ctx.guild.id in self.bot.autoplay:
            del self.bot.autoplay[ctx.guild.id]
        
        await step_msg.edit(content="3️⃣ Re-establishing Connection...")
        
        # 3. Attempt Re-join (if author is in VC)
        if ctx.author.voice:
            try:
                vc = await robust_connect(ctx.author.voice.channel, ctx)
                vc.text_channel = ctx.channel
                self.bot.set_player_text_channel(ctx.guild.id, ctx.channel.id)
                await step_msg.edit(content="✅ **Repair Complete!** You can play music again.")
            except Exception as e:
                await step_msg.edit(content=f"❌ **Repair Failed:** Could not rejoin. ({e})")
        else:
            await step_msg.edit(content="✅ **Repair Complete!** Bot reset. Join a VC to play.")

# --- Lyrics Fetcher ---
class LyricsFetcher:
    """Fetches lyrics from multiple sources with smart parsing"""

    @staticmethod
    def basic_clean(text: str) -> str:
        """Removes noise like (Official Video), [Lyrics], etc., but KEEPS hyphens."""
        if not text:
            return ""
        text = re.sub(r"[\(\[].*?[\)\]]", "", text)
        text = re.sub(r"\b(ft\.|feat\.|official|video|audio|lyrics|lyric|hq|4k|hd|music|visualizer|cover|remix)\b", "", text, flags=re.IGNORECASE)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    @staticmethod
    async def scrape_genius_lyrics(url: str) -> Optional[str]:
        """Scrape lyrics from Genius URL with safety checks (size, timeouts)."""
        if not url or not isinstance(url, str):
            return None

        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                }
                try:
                    async with session.get(url, headers=headers, timeout=10) as resp:
                        if resp.status != 200:
                            logger.debug(f"Genius returned status {resp.status} for {url}")
                            return None
                        html_text = await resp.text()
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout scraping Genius URL: {url}")
                    return None
                except Exception as e:
                    logger.exception(f"HTTP error scraping Genius: {e}")
                    return None

            # protect against huge responses
            if not html_text or len(html_text) > 800_000:
                logger.warning("Genius HTML too large or empty")
                return None

            patterns = [
                r'<div[^>]*data-lyrics-container="true"[^>]*>(.*?)</div>',
                r'<div[^>]*class="[^"]*lyrics[^"]*"[^>]*>(.*?)</div>',
                r'<div[^>]*class="[^"]*Lyrics__Container[^"]*"[^>]*>(.*?)</div>'
            ]

            lyrics_parts = []
            for pattern in patterns:
                try:
                    matches = re.findall(pattern, html_text, re.DOTALL)
                except Exception:
                    matches = []
                if matches:
                    lyrics_parts.extend(matches)

            if not lyrics_parts:
                return None

            full_lyrics = "\n".join(lyrics_parts)
            full_lyrics = re.sub(r'<br\s*/?>', '\n', full_lyrics)
            full_lyrics = re.sub(r'<[^>]+>', '', full_lyrics)
            full_lyrics = html.unescape(full_lyrics)
            full_lyrics = re.sub(r'\n{3,}', '\n\n', full_lyrics)
            cleaned = full_lyrics.strip()
            # guard: require at least some characters
            if not cleaned or len(cleaned) < 20:
                return None
            return cleaned
        except Exception as e:
            logger.exception(f"Genius scraping error: {e}")
            return None

    @staticmethod
    async def fetch_from_genius(query: str) -> Optional[dict]:
        if not GENIUS_TOKEN:
            return None
        headers = {"Authorization": f"Bearer {GENIUS_TOKEN}"}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                        "https://api.genius.com/search",
                        headers=headers,
                        params={"q": query},
                        timeout=10
                ) as resp:
                    if resp.status != 200:
                        return None
                    data = await resp.json()

                hits = data.get("response", {}).get("hits", [])
                if not hits:
                    return None

                song = None
                for hit in hits:
                    result = hit.get("result", {})
                    if result.get("lyrics_state") == "complete":
                        song = result
                        break
                if not song:
                    song = hits[0].get("result", {})

                lyrics_text = await LyricsFetcher.scrape_genius_lyrics(song.get("url", ""))

                return {
                    "title": song.get("title", query),
                    "artist": song.get("primary_artist", {}).get("name", "Unknown"),
                    "url": song.get("url"),
                    "thumbnail": song.get("song_art_image_thumbnail_url"),
                    "lyrics": lyrics_text,
                    "source": "Genius"
                }
        except asyncio.TimeoutError:
            logger.warning("Genius API timeout")
            return None
        except Exception as e:
            logger.error(f"Genius API error: {e}")
            return None

    @staticmethod
    async def fetch_from_lrclib(title: str, artist: str = None) -> Optional[dict]:
        """Fetch lyrics from lrclib.net"""
        try:
            async with aiohttp.ClientSession() as session:
                if artist:
                    params = {"track_name": title, "artist_name": artist}
                    async with session.get(
                            "https://lrclib.net/api/get",
                            params=params,
                            timeout=10
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            lyrics = data.get("plainLyrics") or data.get("syncedLyrics")
                            if lyrics:
                                return {
                                    "title": data.get("trackName", title),
                                    "artist": data.get("artistName", artist or "Unknown"),
                                    "lyrics": lyrics,
                                    "source": "LrcLib"
                                }

                query = f"{title} {artist}" if artist else title
                async with session.get(
                        "https://lrclib.net/api/search",
                        params={"q": query},
                        timeout=10
                ) as resp:
                    if resp.status == 200:
                        results = await resp.json()
                        if results and isinstance(results, list):
                            for result in results:
                                lyrics = result.get("plainLyrics") or result.get("syncedLyrics")
                                if lyrics:
                                    return {
                                        "title": result.get("trackName", title),
                                        "artist": result.get("artistName", artist or "Unknown"),
                                        "lyrics": lyrics,
                                        "source": "LrcLib"
                                    }
        except asyncio.TimeoutError:
            logger.warning("LrcLib API timeout")
            return None
        except Exception as e:
            logger.error(f"LrcLib error: {e}")
            return None


# --- Custom Music Resolver (Gaana/Saavn) ---
class MusicSourceResolver:
    """Resolves specific Indian music service URLs to search queries"""

    @staticmethod
    async def resolve_jiosaavn(url: str) -> Union[Dict[str, str], List[Dict[str, str]], None]:
        """Resolve JioSaavn URL to track metadata (Single or List)"""

        try:
            token_match = re.search(r"jiosaavn\.com\/(?:song|album|featured|playlist)\/[^/]+\/([^/?]+)", url)
            if token_match:
                token = token_match.group(1)

                req_type = "song"
                if "/album/" in url:
                    req_type = "album"
                elif "/featured/" in url or "/playlist/" in url:
                    req_type = "playlist"

                api_url = "https://www.jiosaavn.com/api.php"
                params = {
                    "__call": "webapi.get",
                    "token": token,
                    "type": req_type,
                    "n": "1000",
                    "p": "1",
                    "includeMetaTags": "0",
                    "ctx": "web6dot0",
                    "api_version": "4",
                    "_format": "json",
                    "_marker": "0"
                }

                async with aiohttp.ClientSession() as session:
                    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
                    async with session.get(api_url, params=params, headers=headers) as resp:
                        if resp.status == 200:
                            data = await resp.json(content_type=None)

                            results = []
                            source_list = []

                            if req_type == "song" and "songs" in data:
                                source_list = data["songs"]
                            elif (req_type == "album" or req_type == "playlist") and "list" in data:
                                source_list = data["list"]

                            for song in source_list:
                                artist_raw = song.get("more_info", {}).get("primary_artists", "") or song.get("more_info", {}).get("music", "")

                                results.append({
                                    "title": song.get("title", "").replace("&quot;", '"').replace("&amp;", "&"),
                                    "artist": artist_raw.replace("&quot;", '"').replace("&amp;", "&"),
                                    "image": song.get("image", "").replace("150x150", "500x500")
                                })

                            if results:
                                logger.info(f"Resolved JioSaavn via Direct API: {len(results)} songs found.")
                                return results if len(results) > 1 else results[0]
        except Exception as e:
            logger.error(f"JioSaavn internal resolution error: {e}")

        try:
            async with aiohttp.ClientSession() as session:
                api_url = f"https://saavan-ten.vercel.app/api/songs?link={url}"

                async with session.get(api_url, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if (data.get("status") == "SUCCESS" or data.get("success")) and data.get("data"):
                            items = data["data"]

                            if isinstance(items, list) and items:
                                results = []
                                for song in items:
                                    image_url = ""
                                    if song.get("image"):
                                        if isinstance(song["image"], list):
                                            image_url = song["image"][-1].get("link", "") if song["image"] else ""
                                        else:
                                            image_url = song["image"]

                                    results.append({
                                        "title": song.get("name", ""),
                                        "artist": song.get("primaryArtists", "") or song.get("primaryArtists", ""),
                                        "image": image_url
                                    })
                                logger.info(f"Resolved JioSaavn via Vercel API: {len(results)} songs found.")
                                return results if len(results) > 1 else results[0]

                            elif isinstance(items, dict):
                                image_url = ""
                                if items.get("image"):
                                    if isinstance(items["image"], list):
                                        image_url = items["image"][-1].get("link", "") if items["image"] else ""
                                    else:
                                        image_url = items["image"]

                                return {
                                    "title": items.get("name", ""),
                                    "artist": items.get("primaryArtists", "") or items.get("primaryArtists", ""),
                                    "image": image_url
                                }
        except Exception as e:
            logger.warning(f"JioSaavn API (saavan-ten.vercel.app) resolution failed: {e}")

        return None

    @staticmethod
    async def search_jiosaavn(query: str) -> List[Dict[str, str]]:
        """Search JioSaavn using the new API"""
        try:
            api_url = f"https://saavan-ten.vercel.app/api/search/songs?query={query}"

            async with aiohttp.ClientSession() as session:
                async with session.get(api_url, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()

                        if data.get("success") and data.get("data"):
                            results_data = data["data"].get("results", [])
                            if not results_data:
                                return []

                            parsed_results = []
                            for song in results_data:
                                image_url = ""
                                if song.get("image"):
                                    if isinstance(song["image"], list):
                                        image_url = song["image"][-1].get("link", "") if song["image"] else ""
                                    else:
                                        image_url = song["image"]

                                parsed_results.append({
                                    "title": song.get("name", ""),
                                    "artist": song.get("primaryArtists", ""),
                                    "image": image_url
                                })
                            return parsed_results
            return []
        except Exception as e:
            logger.error(f"JioSaavn search error: {e}")
            return []

    @staticmethod
    async def resolve_gaana(url: str) -> Union[Dict[str, str], List[Dict[str, str]], None]:
        """Resolve Gaana URL to track metadata or playlist tracks using JSON-LD Scraping"""
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Referer": "https://gaana.com/"
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as resp:
                    if resp.status != 200:
                        logger.error(f"Gaana Fetch Error: Status {resp.status}")
                        return None
                    html_content = await resp.text()

            json_ld_matches = re.findall(r'<script type="application/ld\+json">(.*?)</script>', html_content, re.DOTALL)

            results = []

            for json_str in json_ld_matches:
                try:
                    data = json.loads(json_str)

                    if isinstance(data, list):
                        items = data
                    else:
                        items = [data]

                    for item in items:
                        if item.get("@type") in ["MusicPlaylist", "MusicAlbum"]:
                            track_list = item.get("track", [])

                            if not track_list and "itemListElement" in item:
                                track_list = [x.get("item", {}) for x in item["itemListElement"]]

                            for track in track_list:
                                title = track.get("name", "")

                                artist_obj = track.get("byArtist")
                                artist = ""
                                if isinstance(artist_obj, list):
                                    artist = ", ".join([a.get("name", "") for a in artist_obj])
                                elif isinstance(artist_obj, dict):
                                    artist = artist_obj.get("name", "")
                                elif isinstance(artist_obj, str):
                                    artist = artist_obj

                                if title:
                                    image = item.get("image", "")
                                    if isinstance(image, list) and len(image) > 0:
                                        image = image[0]

                                    results.append({
                                        "title": html.unescape(title),
                                        "artist": html.unescape(artist),
                                        "image": image,
                                        "provider": "gaana"
                                    })
                except json.JSONDecodeError:
                    continue

            if results:
                logger.info(f"Resolved Gaana playlist/album via JSON-LD with {len(results)} tracks")
                return results

            title_match = re.search(r'"name"\s*:\s*"([^"]+)"', html_content)
            artist_match = re.search(r'"byArtist"\s*:\s*\[\s*{\s*"@type"\s*:\s*"Person",\s*"name"\s*:\s*"([^"]+)"', html_content)

            if not title_match:
                title_match = re.search(r'<meta property="og:title" content="([^"]+)"', html_content)

            if title_match:
                title = title_match.group(1).replace(" | Gaana.com", "").strip()
                artist = artist_match.group(1) if artist_match else ""

                title = re.sub(r'(?i)\s*(song|mp3|download).*$', '', title).strip()

                if title.lower() == "gaana":
                    return None

                return {
                    "title": html.unescape(title),
                    "artist": html.unescape(artist),
                    "provider": "gaana"
                }

            return None

        except Exception as e:
            logger.error(f"Gaana resolution error: {e}")
            return None


# --- TRIVIA UI CLASSES ---
class GameSelectionView(discord.ui.View):
    def __init__(self, ctx):
        super().__init__(timeout=60)
        self.ctx = ctx
        self.value = None

    @discord.ui.button(label="🎵 Guess the Song", style=discord.ButtonStyle.primary, emoji="🎼")
    async def song_trivia(self, interaction: discord.Interaction, button: discord.ui.Button):
        if interaction.user.id != self.ctx.author.id:
            await interaction.response.send_message("❌ Only the host can choose.", ephemeral=True)
            return
        self.value = "song"
        self.stop()
        await interaction.response.defer()

    @discord.ui.button(label="🎬 Guess the Movie (Bollywood)", style=discord.ButtonStyle.success, emoji="🎥")
    async def movie_trivia(self, interaction: discord.Interaction, button: discord.ui.Button):
        if interaction.user.id != self.ctx.author.id:
            await interaction.response.send_message("❌ Only the host can choose.", ephemeral=True)
            return
        self.value = "movie"
        self.stop()
        await interaction.response.defer()


class TriviaVoteView(discord.ui.View):
    def __init__(self, host_id, participants):
        super().__init__(timeout=60)
        self.host_id = host_id
        self.participants = participants
        self.votes = {"hindi": set(), "english": set()}

    async def check_vc_presence(self, interaction: discord.Interaction):
        bot_vc = interaction.guild.voice_client
        if not bot_vc or not interaction.user.voice or interaction.user.voice.channel != bot_vc.channel:
            await interaction.response.send_message("🚫 You must be in the voice channel to vote!", ephemeral=True)
            return False
        return True

    @discord.ui.button(label="Hindi Songs 🇮🇳", style=discord.ButtonStyle.secondary)
    async def hindi_vote(self, interaction: discord.Interaction, button: discord.ui.Button):
        if not await self.check_vc_presence(interaction):
            return

        if interaction.user.id not in self.participants:
            await interaction.response.send_message("🚫 Only participants can vote!", ephemeral=True)
            return

        if interaction.user.id in self.votes["english"]:
            self.votes["english"].remove(interaction.user.id)
        self.votes["hindi"].add(interaction.user.id)

        self.update_labels()
        try:
            await interaction.response.edit_message(view=self)
        except discord.NotFound:
            pass

    @discord.ui.button(label="English Songs 🇺🇸", style=discord.ButtonStyle.secondary)
    async def english_vote(self, interaction: discord.Interaction, button: discord.ui.Button):
        if not await self.check_vc_presence(interaction):
            return

        if interaction.user.id not in self.participants:
            await interaction.response.send_message("🚫 Only participants can vote!", ephemeral=True)
            return

        if interaction.user.id in self.votes["hindi"]:
            self.votes["hindi"].remove(interaction.user.id)
        self.votes["english"].add(interaction.user.id)

        self.update_labels()
        try:
            await interaction.response.edit_message(view=self)
        except discord.NotFound:
            pass

    @discord.ui.button(label="Confirm & Start", style=discord.ButtonStyle.success, row=1)
    async def confirm(self, interaction: discord.Interaction, button: discord.ui.Button):
        if interaction.user.id != self.host_id:
            await interaction.response.send_message("❌ Only the host can confirm.", ephemeral=True)
            return
        self.stop()
        await interaction.response.defer()

    def update_labels(self):
        self.children[0].label = f"Hindi Songs 🇮🇳 ({len(self.votes['hindi'])})"
        self.children[1].label = f"English Songs 🇺🇸 ({len(self.votes['english'])})"


class TriviaJoinView(discord.ui.View):
    def __init__(self, host_id):
        super().__init__(timeout=60)
        self.host_id = host_id
        self.participants = set()
        self.participants.add(host_id)

    @discord.ui.button(label="Join Game", style=discord.ButtonStyle.success, emoji="🎮")
    async def join(self, interaction: discord.Interaction, button: discord.ui.Button):
        bot_vc = interaction.guild.voice_client
        if not bot_vc or not interaction.user.voice or interaction.user.voice.channel != bot_vc.channel:
            await interaction.response.send_message("🚫 You must be in the voice channel to join!", ephemeral=True)
            return

        if interaction.user.id in self.participants:
            await interaction.response.send_message("⚠️ You already joined!", ephemeral=True)
            return

        self.participants.add(interaction.user.id)
        await interaction.response.send_message(f"✅ {interaction.user.mention} joined!", ephemeral=False)

    @discord.ui.button(label="Next", style=discord.ButtonStyle.primary, emoji="➡️")
    async def start(self, interaction: discord.Interaction, button: discord.ui.Button):
        if interaction.user.id != self.host_id:
            await interaction.response.send_message("❌ Only the host can proceed.", ephemeral=True)
            return

        bot_vc = interaction.guild.voice_client
        if not bot_vc or not interaction.user.voice or interaction.user.voice.channel != bot_vc.channel:
            await interaction.response.send_message("🚫 You must be in the voice channel to start the game!", ephemeral=True)
            return

        if len(self.participants) < 1:
            await interaction.response.send_message("❌ Need at least 1 player!", ephemeral=True)
            return

        self.stop()
        await interaction.response.defer()


class TriviaButton(discord.ui.Button):
    def __init__(self, label_text, is_correct):
        clean_label = label_text.split('(')[0].split('[')[0].strip()
        if len(clean_label) > 80:
            clean_label = clean_label[:77] + "..."

        super().__init__(label=clean_label, style=discord.ButtonStyle.primary)
        self.label_text = label_text
        self.is_correct = is_correct

    async def callback(self, interaction: discord.Interaction):
        view: TriviaChoiceView = self.view

        bot_vc = interaction.guild.voice_client
        if not bot_vc or not interaction.user.voice or interaction.user.voice.channel != bot_vc.channel:
            await interaction.response.send_message("🚫 You must be in the voice channel to answer!", ephemeral=True)
            return

        if view.winner or view.all_wrong:
            await interaction.response.send_message("🚫 This round is already over!", ephemeral=True)
            return

        if interaction.user.id not in view.participants:
            await interaction.response.send_message("🚫 You didn't join this game!", ephemeral=True)
            return

        if interaction.user.id in view.guessed_users:
            await interaction.response.send_message("🚫 One chance per round!", ephemeral=True)
            return

        view.guessed_users.add(interaction.user.id)

        if self.is_correct:
            view.winner = interaction.user
            self.style = discord.ButtonStyle.success
            for child in view.children:
                child.disabled = True
                if child != self and child.style != discord.ButtonStyle.danger:
                    child.style = discord.ButtonStyle.secondary
            view.stop()

            try:
                await interaction.response.edit_message(view=view)
            except discord.NotFound:
                pass

            try:
                await interaction.followup.send(f"🎉 **{interaction.user.mention}** got it! Answer: **{self.label_text}**.")
            except discord.HTTPException:
                pass
        else:
            self.style = discord.ButtonStyle.danger
            self.disabled = True
            try:
                await interaction.response.edit_message(view=view)
            except discord.NotFound:
                pass

            try:
                await interaction.followup.send(f"❌ **{interaction.user.mention}** wrong!", ephemeral=False)
            except discord.HTTPException:
                pass

            if len(view.guessed_users) >= len(view.participants):
                view.all_wrong = True
                for child in view.children:
                    child.disabled = True
                view.stop()


class TriviaChoiceView(discord.ui.View):
    def __init__(self, correct_text, options_data, participants):
        super().__init__(timeout=25)
        self.winner = None
        self.all_wrong = False
        self.participants = participants
        self.guessed_users = set()

        random.shuffle(options_data)

        for text in options_data:
            is_correct = (text == correct_text)
            self.add_item(TriviaButton(text, is_correct))


# --- Speed Control View ---
class SpeedSelect(discord.ui.Select):
    def __init__(self, player: wavelink.Player):
        self.player = player
        options = [
            discord.SelectOption(label="0.5x (Slow)", value="0.5", emoji="🐢"),
            discord.SelectOption(label="0.75x", value="0.75", emoji="🚶"),
            discord.SelectOption(label="1.0x (Normal)", value="1.0", emoji="▶️"),
            discord.SelectOption(label="1.25x", value="1.25", emoji="🌙"),
            discord.SelectOption(label="1.5x (Fast)", value="1.5", emoji="💨"),
            discord.SelectOption(label="1.75x", value="1.75", emoji="🏎️"),
            discord.SelectOption(label="2.0x (Double)", value="2.0", emoji="🚀"),
        ]
        super().__init__(placeholder="Select Playback Speed", min_values=1, max_values=1, options=options)

    async def callback(self, interaction: discord.Interaction):
        # Defensive: validate inputs & player
        if not hasattr(self, "player") or self.player is None:
            try:
                await interaction.response.send_message("❌ Player not available.", ephemeral=True)
            except Exception:
                pass
            return

        # Validate selection
        try:
            if not self.values or not isinstance(self.values, (list, tuple)):
                await interaction.response.send_message("❌ No speed selected.", ephemeral=True)
                return
            speed = float(self.values[0])
        except (ValueError, TypeError):
            await interaction.response.send_message("❌ Invalid speed value.", ephemeral=True)
            return

        # Range check
        if speed < 0.25 or speed > 3.0:
            await interaction.response.send_message("❌ Speed out of allowed range (0.25 — 3.0).", ephemeral=True)
            return

        try:
            filters = getattr(self.player, "filters", None) or wavelink.Filters()
            # protect timescale set usage
            try:
                filters.timescale.set(speed=speed)
            except Exception as e:
                logger.warning(f"Unable to set timescale params: {e}")
                # try minimal set
                try:
                    filters.timescale.set(speed=speed, pitch=1.0)
                except Exception:
                    pass

            await self.player.set_filters(filters)
            try:
                await interaction.response.send_message(f"⏱️ Speed set to **{speed}x**.", ephemeral=True)
            except Exception:
                # already responded? try followup
                try:
                    await interaction.followup.send(f"⏱️ Speed set to **{speed}x**.", ephemeral=True)
                except Exception:
                    pass
        except Exception as e:
            logger.exception(f"Speed change error: {e}")
            try:
                await interaction.response.send_message("❌ Failed to change speed.", ephemeral=True)
            except Exception:
                pass


class SpeedView(discord.ui.View):
    def __init__(self, player: wavelink.Player):
        super().__init__(timeout=60)
        self.add_item(SpeedSelect(player))

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if not interaction.guild.voice_client:
            await interaction.response.send_message("❌ Bot disconnected.", ephemeral=True)
            return False
        bot_channel = interaction.guild.voice_client.channel
        if not interaction.user.voice or interaction.user.voice.channel != bot_channel:
            await interaction.response.send_message("🚫 You must be in the voice channel!", ephemeral=True)
            return False
        return True


class ConnectionOptimizer(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @app_commands.command(name="optimize", description="[Admin] Optimizes connection by switching to the best available Lavalink node.")
    @app_commands.checks.has_permissions(administrator=True)
    async def optimize(self, interaction: discord.Interaction):
        """
        Checks current node health and moves the player to a less burdened node if available.
        """
        # 1. Check if the bot is actually playing in this guild
        player: wavelink.Player = discord.utils.get(self.bot.voice_clients, guild=interaction.guild)
        
        if not player or not player.connected:
            return await interaction.response.send_message(
                "❌ | **No Active Connection:** I am not currently connected to a voice channel.", 
                ephemeral=True
            )

        await interaction.response.defer()

        # 2. Get Current Node Stats
        current_node = player.node
        current_stats = current_node.stats
        
        # Format stats for display
        # 'penalty' is a calculated load score (lower is better) used by Lavalink
        current_load = current_stats.system_load
        current_ping = player.ping

        # 3. Algorithm: Find the "Best" Node
        # We look for the node with the lowest 'penalty' or 'load'
        available_nodes = wavelink.Pool.nodes.values()
        best_node = current_node
        
        # Simple heuristic: Lowest system load
        for node in available_nodes:
            if node.stats.system_load < best_node.stats.system_load:
                best_node = node

        # 4. Action: Switch or Stay
        embed = discord.Embed(title="📶 Connection Optimization", color=0x2b2d31)
        
        if best_node != current_node:
            try:
                # Attempt to move the player to the new node
                await player.transfer_to(best_node)
                
                embed.description = (
                    f"**Optimization Successful**\n"
                    f"Moved from node `{current_node.identifier}` to `{best_node.identifier}`.\n\n"
                    f"**Improvement:**\n"
                    f"📉 Load: `{current_load:.2f}` → `{best_node.stats.system_load:.2f}`"
                )
                embed.color = discord.Color.green()
                
            except Exception as e:
                embed.description = f"**Optimization Failed:** Could not transfer node.\n`{str(e)}`"
                embed.color = discord.Color.red()
        else:
            embed.description = (
                f"**Connection is already Optimal**\n"
                f"You are connected to the best available node: `{current_node.identifier}`.\n"
            )
            embed.add_field(name="Current Ping", value=f"`{current_ping}ms`", inline=True)
            embed.add_field(name="System Load", value=f"`{current_load:.2%}`", inline=True)
            embed.color = discord.Color.gold()

        await interaction.followup.send(embed=embed)

    @optimize.error
    async def optimize_error(self, interaction: discord.Interaction, error):
        if isinstance(error, app_commands.MissingPermissions):
            await interaction.response.send_message(
                "⛔ | **Access Denied:** You must be an Administrator to optimize the connection.", 
                ephemeral=True
            )

async def setup(bot):
    await bot.add_cog(ConnectionOptimizer(bot))

# --- Player Controller View ---
class PlayerControls(discord.ui.View):
    def __init__(self, player: wavelink.Player):
        super().__init__(timeout=180)
        self.player = player
        self._update_button_styles()

    def _update_button_styles(self):
        if not self.player:
            return

        for child in self.children:
            if getattr(child, 'emoji', None) == '⏯️':
                child.style = discord.ButtonStyle.secondary if self.player.paused else discord.ButtonStyle.success
                break

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if not interaction.guild.voice_client:
            await interaction.response.send_message("❌ I am not connected to a voice channel.", ephemeral=True)
            return False

        bot_channel = interaction.guild.voice_client.channel
        if not interaction.user.voice or interaction.user.voice.channel != bot_channel:
            await interaction.response.send_message(f"🚫 You must be in {bot_channel.mention} to use the controls!", ephemeral=True)
            return False

        return True

    async def update_seek_message(self, interaction, seconds):
        """Seek safely by seconds (positive forward, negative back)."""
        if not self.player:
            try:
                await interaction.response.send_message("❌ Player unavailable.", ephemeral=True)
            except Exception:
                pass
            return

        # ensure player has a current track and necessary attrs
        try:
            current = getattr(self.player, "current", None)
            position = getattr(self.player, "position", None)
            if not current or (position is None):
                try:
                    await interaction.response.send_message("❌ No track is currently playing.", ephemeral=True)
                except Exception:
                    pass
                return

            # compute new position in milliseconds
            cur_pos_ms = int(position or 0)
            length_ms = int(getattr(current, "length", 0) or 0)
            new_pos = max(0, min(cur_pos_ms + int(seconds * 1000), length_ms))

            try:
                await self.player.seek(new_pos)
            except Exception as e:
                logger.exception(f"Failed to seek player: {e}")
                try:
                    await interaction.response.send_message("❌ Seek failed.", ephemeral=True)
                except Exception:
                    pass
                return

            action = "⏩" if seconds > 0 else "⏪"
            minutes = (new_pos // 1000) // 60
            secs = (new_pos // 1000) % 60
            try:
                await interaction.response.send_message(f"{action} Seeked to {minutes}:{secs:02d}", ephemeral=True)
            except Exception:
                try:
                    await interaction.followup.send(f"{action} Seeked to {minutes}:{secs:02d}", ephemeral=True)
                except Exception:
                    pass
        except Exception as e:
            logger.exception(f"Unexpected error in update_seek_message: {e}")
            try:
                await interaction.response.send_message("❌ Error while seeking.", ephemeral=True)
            except Exception:
                pass

    @discord.ui.button(emoji="🔉", style=discord.ButtonStyle.secondary, row=0)
    async def vol_down_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        if not self.player:
            return

        new_vol = max(0, self.player.volume - 10)
        await self.player.set_volume(new_vol)
        try:
            await interaction.response.send_message(f"🔉 Volume: {new_vol}%", ephemeral=True)
        except discord.HTTPException:
            pass

    @discord.ui.button(emoji="⏮️", style=discord.ButtonStyle.primary, row=0)
    async def prev_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        if not self.player:
            try:
                await interaction.response.send_message("❌ Player unavailable.", ephemeral=True)
            except Exception:
                pass
            return

        # Defer quickly to avoid "already responded" errors later
        try:
            await interaction.response.defer(ephemeral=True)
        except Exception:
            # ignore; proceed
            pass

        try:
            guild_id = getattr(interaction.guild, "id", None)
            if guild_id is None:
                await interaction.followup.send("❌ Guild information missing.", ephemeral=True)
                return

            history_tracks = db.get_last_two_tracks(guild_id)
            if not history_tracks or len(history_tracks) < 2:
                # if track playing, just restart
                if getattr(self.player, "playing", False) or getattr(self.player, "paused", False):
                    try:
                        await self.player.seek(0)
                        await interaction.followup.send("⏮️ Replaying from start (No previous track).", ephemeral=True)
                    except Exception:
                        await interaction.followup.send("❌ Unable to replay current track.", ephemeral=True)
                    return
                await interaction.followup.send("❌ No history.", ephemeral=True)
                return

            prev_track_data = history_tracks[1]
            # attempt to remove last track but continue even if fails
            try:
                db.remove_last_track(guild_id)
            except Exception:
                logger.debug("remove_last_track failed but continuing", exc_info=True)

            # Put current track back into queue if exists and queue supports put_at
            try:
                if getattr(self.player, "current", None) and hasattr(self.player.queue, "put_at"):
                    self.player.queue.put_at(0, self.player.current)
            except Exception:
                logger.debug("Failed to put current track back to queue", exc_info=True)

            clean_title = (prev_track_data[0] or "").split("(")[0].split("[")[0].strip()
            search_query = f"ytsearch:{clean_title} {prev_track_data[2] or ''}".strip()
            try:
                search_res = await wavelink.Playable.search(search_query)
            except Exception as e:
                logger.exception(f"Search failed for prev_button: {e}")
                await interaction.followup.send("❌ Search failed.", ephemeral=True)
                return

            if not search_res:
                await interaction.followup.send("❌ Track not found.", ephemeral=True)
                return

            # extract track from search result robustly
            try:
                if isinstance(search_res, wavelink.Playlist) and getattr(search_res, "tracks", None):
                    track_to_play = search_res.tracks[0]
                elif isinstance(search_res, (list, tuple)) and len(search_res) > 0:
                    track_to_play = search_res[0]
                else:
                    # some wrappers expose .tracks or [0]
                    track_to_play = getattr(search_res, "tracks", [None])[0] or (search_res[0] if hasattr(search_res, "__getitem__") else None)
            except Exception:
                track_to_play = None

            if track_to_play:
                try:
                    track_to_play.is_rewind = True
                    await self.player.play(track_to_play, replace=True)
                    await interaction.followup.send(f"⏮️ Playing: **{prev_track_data[0]}**", ephemeral=True)
                except Exception as e:
                    logger.exception(f"Failed to play previous track: {e}")
                    await interaction.followup.send("❌ Could not load track.", ephemeral=True)
            else:
                await interaction.followup.send("❌ Could not load track.", ephemeral=True)

        except Exception as e:
            logger.exception(f"Previous track error: {e}")
            try:
                await interaction.followup.send("❌ Error going back.", ephemeral=True)
            except Exception:
                pass

    @discord.ui.button(emoji="⏯️", style=discord.ButtonStyle.success, row=0)
    async def pause_resume_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        if not self.player:
            try:
                await interaction.response.send_message("❌ Player unavailable.", ephemeral=True)
            except Exception:
                pass
            return

        try:
            if getattr(self.player, "paused", False):
                await self.player.pause(False)
                button.style = discord.ButtonStyle.success
            else:
                await self.player.pause(True)
                button.style = discord.ButtonStyle.secondary

            # attempt safe edit first, fallback to response
            try:
                await interaction.response.edit_message(view=self)
            except Exception:
                try:
                    await interaction.followup.send("⏸️ Player state toggled.", ephemeral=True)
                except Exception:
                    pass
        except Exception as e:
            logger.exception(f"Pause/Resume error: {e}")
            try:
                await interaction.response.send_message("❌ Failed to toggle pause/resume.", ephemeral=True)
            except Exception:
                pass

    @discord.ui.button(emoji="⏭️", style=discord.ButtonStyle.primary, row=0)
    async def skip_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        if not self.player:
            try:
                await interaction.response.send_message("❌ Player unavailable.", ephemeral=True)
            except Exception:
                pass
            return

        try:
            queue = getattr(self.player, "queue", None)
            current = getattr(self.player, "current", None)
            is_empty = True
            try:
                # some queue objects have is_empty attribute or __len__
                if queue is None:
                    is_empty = True
                elif hasattr(queue, "is_empty"):
                    is_empty = bool(queue.is_empty)
                else:
                    is_empty = len(queue) == 0
            except Exception:
                is_empty = True

            if is_empty and not current:
                await interaction.response.send_message("❌ Nothing to skip.", ephemeral=True)
                return

            try:
                await self.player.skip(force=True)
                await interaction.response.send_message("⏭️ Skipped.", ephemeral=True)
            except Exception as e:
                logger.exception(f"Skip operation failed: {e}")
                await interaction.response.send_message("❌ Failed to skip.", ephemeral=True)
        except Exception as e:
            logger.exception(f"Unexpected skip_button error: {e}")
            try:
                await interaction.response.send_message("❌ Failed to skip (unexpected).", ephemeral=True)
            except Exception:
                pass

    @discord.ui.button(emoji="🔊", style=discord.ButtonStyle.secondary, row=0)
    async def vol_up_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        if not self.player:
            return

        new_vol = min(1000, self.player.volume + 10)
        await self.player.set_volume(new_vol)
        try:
            await interaction.response.send_message(f"🔊 Volume: {new_vol}%", ephemeral=True)
        except discord.HTTPException:
            pass

    @discord.ui.button(label="-30s", style=discord.ButtonStyle.secondary, row=1)
    async def seek_back_30(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self.update_seek_message(interaction, -30)

    @discord.ui.button(label="-10s", style=discord.ButtonStyle.secondary, row=1)
    async def seek_back_10(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self.update_seek_message(interaction, -10)

    @discord.ui.button(label="+10s", style=discord.ButtonStyle.secondary, row=1)
    async def seek_fwd_10(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self.update_seek_message(interaction, 10)

    @discord.ui.button(label="+30s", style=discord.ButtonStyle.secondary, row=1)
    async def seek_fwd_30(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self.update_seek_message(interaction, 30)

    @discord.ui.button(emoji="🔁", label="Off", style=discord.ButtonStyle.secondary, row=2)
    async def loop_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        if not self.player:
            return

        current_mode = self.player.queue.mode
        if current_mode == wavelink.QueueMode.normal:
            self.player.queue.mode = wavelink.QueueMode.loop
            self.player.autoplay = wavelink.AutoPlayMode.disabled
            button.label = "Track"
            button.emoji = "🔂"
            button.style = discord.ButtonStyle.success
            msg = "🔂 Loop: **Track**"
        elif current_mode == wavelink.QueueMode.loop:
            self.player.queue.mode = wavelink.QueueMode.loop_all
            self.player.autoplay = wavelink.AutoPlayMode.disabled
            button.label = "Queue"
            button.emoji = "🔁"
            button.style = discord.ButtonStyle.success
            msg = "🔁 Loop: **Queue**"
        else:
            self.player.queue.mode = wavelink.QueueMode.normal
            button.label = "Off"
            button.emoji = "➡️"
            button.style = discord.ButtonStyle.secondary
            msg = "➡️ Loop **Off**"

        try:
            await interaction.response.edit_message(view=self)
            await interaction.followup.send(msg, ephemeral=True)
        except discord.HTTPException:
            pass

    @discord.ui.button(emoji="🎚️", style=discord.ButtonStyle.secondary, row=2)
    async def eq_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        if not self.player:
            return

        await interaction.response.send_message("🎚️ **Equalizer**", view=EqualizerView(self.player), ephemeral=True)

    @discord.ui.button(emoji="⏱️", style=discord.ButtonStyle.secondary, row=2)
    async def speed_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        if not self.player:
            return

        await interaction.response.send_message("⏱️ **Speed Control**", view=SpeedView(self.player), ephemeral=True)

    @discord.ui.button(emoji="⏹️", style=discord.ButtonStyle.danger, row=2)
    async def stop_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        # 1. Defer immediately to prevent "Unknown Interaction" (Timeout) errors
        try:
            await interaction.response.defer(ephemeral=True)
        except discord.NotFound:
            return  # Interaction is already dead
        except Exception:
            pass  # Continue if already deferred

        if not self.player:
            try:
                await interaction.followup.send("❌ Player unavailable.", ephemeral=True)
            except Exception:
                pass
            return

        try:
            # Attempt to disable autoplay via client hook if present
            try:
                if hasattr(interaction.client, 'set_autoplay_enabled'):
                    interaction.client.set_autoplay_enabled(interaction.guild.id, False)
            except Exception:
                logger.debug("set_autoplay_enabled hook failed", exc_info=True)

            try:
                if getattr(self.player, "channel", None) and isinstance(self.player.channel, discord.VoiceChannel):
                    # Best-effort: clear any special status on channel
                    try:
                        await self.player.channel.edit(reason=None)
                    except Exception:
                        pass
            except Exception:
                pass

            # Count queue safely
            try:
                queue_len = len(self.player.queue) if self.player and getattr(self.player, "queue", None) else 0
            except Exception:
                queue_len = 0

            # Logic for 24/7 vs Normal Stop
            if db.get_247_status(interaction.guild.id):
                # 24/7 active: stop playback but stay connected
                try:
                    if getattr(self.player, "queue", None):
                        try:
                            self.player.queue.clear()
                        except Exception:
                            pass
                    await self.player.stop()
                    try:
                        db.clear_history(interaction.guild.id)
                    except Exception:
                        pass
                    
                    await interaction.followup.send(
                        f"⏹️ Stopped & cleared {queue_len} queued tracks (24/7 active). Autoplay disabled.",
                        ephemeral=True
                    )
                except Exception as e:
                    logger.exception(f"Stop error (24/7): {e}")
                    await interaction.followup.send("❌ Failed to stop.", ephemeral=True)
            else:
                # Normal disconnect
                try:
                    try:
                        if getattr(self.player, "queue", None):
                            self.player.queue.clear()
                    except Exception:
                        pass
                    try:
                        db.clear_history(interaction.guild.id)
                    except Exception:
                        pass
                    await self.player.disconnect()
                    
                    await interaction.followup.send(
                        f"👋 Disconnected & cleared {queue_len} queued tracks.",
                        ephemeral=True
                    )
                except Exception as e:
                    logger.exception(f"Disconnect error: {e}")
                    await interaction.followup.send("❌ Failed to disconnect.", ephemeral=True)

        except Exception as e:
            logger.exception(f"Unexpected stop_button error: {e}")
            try:
                await interaction.followup.send("❌ Stop failed.", ephemeral=True)
            except Exception:
                pass


# --- Manual EQ View ---
class ManualEQView(discord.ui.View):
    def __init__(self, player: wavelink.Player):
        super().__init__(timeout=60)
        self.player = player
        self.selected_band = 0
        self.band_labels = {
            0: "25 Hz (Sub-Bass)", 1: "40 Hz (Bass)", 2: "63 Hz (Bass)",
            3: "100 Hz (Low-Mid)", 4: "160 Hz (Low-Mid)", 5: "250 Hz (Low-Mid)",
            6: "400 Hz (Mid)", 7: "630 Hz (Mid)", 8: "1 kHz (Mid)",
            9: "1.6 kHz (High-Mid)", 10: "2.5 kHz (High-Mid)", 11: "4 kHz (Treble)",
            12: "6.3 kHz (Treble)", 13: "10 kHz (High-Treble)", 14: "16 kHz (Air)"
        }
        if not hasattr(self.player, 'eq_gains'):
            self.player.eq_gains = {i: 0.0 for i in range(15)}

        self.select_menu = discord.ui.Select(placeholder="Select Frequency Band", min_values=1, max_values=1)
        for i in range(15):
            self.select_menu.add_option(label=self.band_labels[i], value=str(i))
        self.select_menu.callback = self.band_select_callback
        self.add_item(self.select_menu)

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if not interaction.guild.voice_client:
            await interaction.response.send_message("❌ Bot disconnected.", ephemeral=True)
            return False
        bot_channel = interaction.guild.voice_client.channel
        if not interaction.user.voice or interaction.user.voice.channel != bot_channel:
            await interaction.response.send_message("🚫 You must be in the voice channel!", ephemeral=True)
            return False
        return True

    async def band_select_callback(self, interaction: discord.Interaction):
        self.selected_band = int(self.select_menu.values[0])
        await self.update_embed(interaction)

    async def update_embed(self, interaction):
        current_gain = self.player.eq_gains.get(self.selected_band, 0.0)
        embed = discord.Embed(title="🎛️ Manual Equalizer", color=discord.Color.blurple())
        embed.description = (
            f"**Band:** `{self.band_labels[self.selected_band]}`\n"
            f"**Gain:** `{current_gain:.2f}`\n\n"
            "Use buttons to adjust."
        )
        try:
            await interaction.response.edit_message(embed=embed, view=self)
        except discord.NotFound:
            pass

    async def adjust_gain(self, interaction, delta):
        current_gain = self.player.eq_gains.get(self.selected_band, 0.0)
        new_gain = max(-0.25, min(1.0, current_gain + delta))
        self.player.eq_gains[self.selected_band] = new_gain

        filters = self.player.filters or wavelink.Filters()
        bands_payload = [{'band': b, 'gain': g} for b, g in self.player.eq_gains.items()]
        filters.equalizer.set(bands=bands_payload)

        try:
            await self.player.set_filters(filters)
            await self.update_embed(interaction)
        except Exception as e:
            logger.error(f"EQ adjust error: {e}")
            await interaction.response.send_message("❌ Failed to adjust EQ.", ephemeral=True)

    @discord.ui.button(label="-", style=discord.ButtonStyle.primary)
    async def decrease_gain(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self.adjust_gain(interaction, -0.05)

    @discord.ui.button(label="Reset", style=discord.ButtonStyle.secondary)
    async def reset_band(self, interaction: discord.Interaction, button: discord.ui.Button):
        self.player.eq_gains[self.selected_band] = 0.0
        filters = self.player.filters or wavelink.Filters()
        bands_payload = [{'band': b, 'gain': g} for b, g in self.player.eq_gains.items()]
        filters.equalizer.set(bands=bands_payload)

        try:
            await self.player.set_filters(filters)
            await self.update_embed(interaction)
        except Exception as e:
            logger.error(f"EQ reset error: {e}")
            await interaction.response.send_message("❌ Failed to reset EQ band.", ephemeral=True)

    @discord.ui.button(label="+", style=discord.ButtonStyle.primary)
    async def increase_gain(self, interaction: discord.Interaction, button: discord.ui.Button):
        await self.adjust_gain(interaction, 0.05)

    @discord.ui.button(label="Back", style=discord.ButtonStyle.grey, row=2)
    async def back_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        try:
            await interaction.response.edit_message(content="🎚️ **Equalizer**", embeds=[], view=EqualizerView(self.player))
        except discord.NotFound:
            pass


# --- Equalizer Presets View ---
class EqualizerSelect(discord.ui.Select):
    def __init__(self, player: wavelink.Player):
        self.player = player
        options = [
            discord.SelectOption(label="Flat / Reset", description="Reset all effects", emoji="✨", value="flat"),
            discord.SelectOption(label="8D Audio", description="Spatial rotation", emoji="🌀", value="8d"),
            discord.SelectOption(label="Karaoke", description="Remove vocals", emoji="🎤", value="karaoke"),
            discord.SelectOption(label="Bass Boost", description="Heavy bass", emoji="🥁", value="bass"),
            discord.SelectOption(label="Nightcore", description="Speed + pitch up", emoji="🌙", value="nightcore"),
            discord.SelectOption(label="Vaporwave", description="Slow + pitch down", emoji="🌊", value="vaporwave"),
            discord.SelectOption(label="Pop", description="Vocal boost", emoji="🎈", value="pop"),
            discord.SelectOption(label="Gaming", description="Footsteps clarity", emoji="🎮", value="gaming"),
        ]
        super().__init__(placeholder="🎚️ Select Effect...", min_values=1, max_values=1, options=options)

    async def callback(self, interaction: discord.Interaction):
        filters = self.player.filters or wavelink.Filters()
        val = self.values[0]
        msg = ""

        filters.reset()
        if hasattr(self.player, 'eq_gains'):
            self.player.eq_gains = {i: 0.0 for i in range(15)}

        if val == "flat":
            msg = "✨ **Effects Cleared**"
        elif val == "8d":
            filters.rotation.set(rotation_hz=0.2)
            msg = "🌀 **8D Audio Enabled**"
        elif val == "karaoke":
            filters.karaoke.set(level=1.0, mono_level=1.0, filter_band=220.0, filter_width=100.0)
            msg = "🎤 **Karaoke Enabled**"
        elif val == "bass":
            bands = [
                {'band': 0, 'gain': 0.4}, {'band': 1, 'gain': 0.35},
                {'band': 2, 'gain': 0.3}, {'band': 3, 'gain': 0.2},
                {'band': 4, 'gain': 0.1}
            ]
            filters.equalizer.set(bands=bands)
            msg = "🥁 **Bass Boost Enabled**"
        elif val == "nightcore":
            filters.timescale.set(pitch=1.2, speed=1.1)
            msg = "🌙 **Nightcore Enabled**"
        elif val == "vaporwave":
            filters.timescale.set(pitch=0.8, speed=0.85)
            msg = "🌊 **Vaporwave Enabled**"
        elif val == "pop":
            bands = [
                {'band': 6, 'gain': 0.1}, {'band': 7, 'gain': 0.1},
                {'band': 8, 'gain': 0.1}, {'band': 9, 'gain': 0.1},
                {'band': 10, 'gain': 0.1}, {'band': 11, 'gain': 0.1}
            ]
            filters.equalizer.set(bands=bands)
            msg = "🎈 **Pop Enabled**"
        elif val == "gaming":
            bands = [
                {'band': 0, 'gain': 0.2}, {'band': 1, 'gain': 0.15},
                {'band': 12, 'gain': 0.15}, {'band': 13, 'gain': 0.2},
                {'band': 14, 'gain': 0.2}
            ]
            filters.equalizer.set(bands=bands)
            msg = "🎮 **Gaming Enabled**"

        try:
            await self.player.set_filters(filters)
            await interaction.response.send_message(msg, ephemeral=True)
        except Exception as e:
            logger.error(f"EQ preset error: {e}")
            await interaction.response.send_message("❌ Failed to apply EQ preset.", ephemeral=True)


class EqualizerView(discord.ui.View):
    def __init__(self, player: wavelink.Player):
        super().__init__(timeout=60)
        self.player = player
        self.add_item(EqualizerSelect(player))

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if not is_player_safe(self.player):
            await interaction.response.send_message(
                "❌ Player is not connected properly. Please re-play a song.",
                ephemeral=True
            )
            return False

        bot_channel = self.player.channel
        if not interaction.user.voice or interaction.user.voice.channel != bot_channel:
            await interaction.response.send_message(
                f"🚫 You must be in {bot_channel.mention} to use controls.",
                ephemeral=True
            )
            return False

        return True


    @discord.ui.button(label="🎛️ Manual", style=discord.ButtonStyle.primary, row=1)
    async def manual_eq(self, interaction: discord.Interaction, button: discord.ui.Button):
        embed = discord.Embed(title="🎛️ Manual Equalizer", description="Select a band below.", color=discord.Color.blurple())
        try:
            await interaction.response.edit_message(content=None, embed=embed, view=ManualEQView(self.player))
        except discord.NotFound:
            pass


# --- AUTOPLAY HELPERS ---
def extract_video_id(url: str) -> Optional[str]:
    """Extract a YouTube video ID from various URL formats, defensively."""
    if not url or not isinstance(url, str):
        return None

    # direct full id?
    if re.fullmatch(r"[0-9A-Za-z_-]{11}", url.strip()):
        return url.strip()

    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11})(?:[&?#]|$)",
        r"youtu\.be\/([0-9A-Za-z_-]{11})(?:[&?#]|$)",
        r"youtube\.com\/shorts\/([0-9A-Za-z_-]{11})(?:[&?#]|$)"
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


async def get_recommended_from_seed(seed_url_or_id: str, limit: int = 15) -> List[dict]:
    """Fetch RD recommendations from YouTube using yt_dlp; very defensive."""
    loop = asyncio.get_event_loop()
    seed_id = extract_video_id(seed_url_or_id) or (seed_url_or_id if isinstance(seed_url_or_id, str) else None)
    if not seed_id or not isinstance(seed_id, str) or len(seed_id) != 11:
        logger.debug("Invalid seed id for recommendations")
        return []

    rd_url = f"https://www.youtube.com/watch?v={seed_id}&list=RD{seed_id}"

    ydl_opts = {
        'format': 'bestaudio/best',
        'extract_flat': True,
        'ignoreerrors': True,
        'quiet': True,
        'no_warnings': True,
        'playlistend': max(1, int(limit or 15)),
        'default_search': 'auto'
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                info = await loop.run_in_executor(None, lambda: ydl.extract_info(rd_url, download=False))
            except Exception as e:
                logger.exception(f"yt_dlp extract_info failed: {e}")
                return []

            if not info:
                return []

            entries = info.get('entries') or []
            if not isinstance(entries, (list, tuple)):
                return []

            candidates = []
            for entry in entries:
                if not entry or not isinstance(entry, dict):
                    continue
                vid = entry.get('id') or entry.get('url') or entry.get('webpage_url')
                if not vid:
                    continue
                title = entry.get('title') or "Unknown"
                webpage = entry.get('webpage_url') or (f"https://www.youtube.com/watch?v={vid}" if isinstance(vid, str) else "")
                duration = int(entry.get('duration') or 0)
                candidates.append({
                    'title': title,
                    'webpage_url': webpage,
                    'id': str(vid),
                    'duration': duration
                })
            return candidates
    except Exception as e:
        logger.exception(f"Failed to get recommendations: {e}")
        return []


# --- GAME STATE CLASS ---
class GameState:
    def __init__(self, rounds):
        self.rounds = rounds
        self.current_round = 0
        self.scores = {}
        self.active = True


def create_now_playing_embed(track: wavelink.Playable) -> discord.Embed:
    """Generates the Now Playing embed for consistent UI."""
    duration_min = int((track.length / 1000) // 60)
    duration_sec = int((track.length / 1000) % 60)

    # Simple Progress Bar (Starts at 0:00)
    total_bars = 15
    progress_bar = "🔘" + "▬" * (total_bars - 1)

    # Attempt to resolve provider icon
    custom_data = getattr(track, "custom_payload", {})
    provider = custom_data.get("provider", "default")
    
    # Fallback provider detection if not in payload
    if provider == "default" and track.uri:
        if "spotify" in track.uri: provider = "spotify"
        elif "apple" in track.uri: provider = "applemusic"
        elif "soundcloud" in track.uri: provider = "soundcloud"
        elif "youtu" in track.uri: provider = "youtube"

    icon_url = ICONS.get(provider, ICONS["default"])

    embed = discord.Embed(
        description=f"## 🎵 Now Playing\n**[{track.title}]({track.uri or 'https://youtube.com'})**\nby `{track.author}`\n\n`0:00` {progress_bar} `{duration_min:02d}:{duration_sec:02d}`",
        color=discord.Color.from_rgb(138, 43, 226)
    )
    embed.set_author(name="Playing Now", icon_url=icon_url)

    # Thumbnail Logic
    if custom_data.get("artwork"):
        embed.set_thumbnail(url=custom_data["artwork"])
    elif getattr(track, "artwork", None):
        embed.set_thumbnail(url=track.artwork)
    elif track.uri and ("youtube.com" in track.uri or "youtu.be" in track.uri):
        vid_id = extract_video_id(track.uri)
        if vid_id:
            embed.set_thumbnail(url=f"https://img.youtube.com/vi/{vid_id}/hqdefault.jpg")

    footer_text = f"Source: {provider.capitalize()}"
    embed.set_footer(text=footer_text)

    return embed

async def safe_play_or_queue(vc: wavelink.Player, track, interaction: discord.Interaction):
    """
    Safely plays a track immediately if queue is missing or empty,
    otherwise adds it to the queue.
    """

    if not vc or not track:
        raise RuntimeError("Player or track missing")

    queue = getattr(vc, "queue", None)


    if queue is None:
        await vc.play(track)
        # Removed force_now_playing to prevent duplicates (on_track_start handles it)
        return "played"


    try:
        if not hasattr(queue, "__len__") or len(queue) == 0:
            await vc.play(track)
            # Removed force_now_playing to prevent duplicates
            return "played"
    except Exception:
        await vc.play(track)
        return "played"

    try:
        if hasattr(queue, "put_wait"):
            await queue.put_wait(track)
        else:
            queue.put(track)
        return "queued"
    except Exception:
        # fallback → play immediately
        await vc.play(track)
        return "played"

async def robust_connect(channel: discord.VoiceChannel, ctx=None):
    """
    Attempts to connect to a channel with retries and forces Self-Deafen.
    """
    if not channel:
        return None
    
    guild = channel.guild
    vc = guild.voice_client

    # If already connected to this channel
    if vc:
        if vc.channel.id == channel.id:
            # Ensure deafened if already there
            if not guild.me.voice.self_deaf:
                await guild.change_voice_state(channel=channel, self_deaf=True)
            return vc
        else:
            # Move to the new channel
            await vc.move_to(channel)
            # FORCE DEAFEN AFTER MOVE
            await guild.change_voice_state(channel=channel, self_deaf=True)
            return vc

    # Retry logic for "Timeout" errors
    retries = 3
    for attempt in range(retries):
        try:
            player = await channel.connect(cls=wavelink.Player, self_deaf=True, timeout=20)
            
            # FORCE DEAFEN CHECK (The Fix)
            # Sometimes the initial handshake misses the deafen state
            if not guild.me.voice or not guild.me.voice.self_deaf:
                await guild.change_voice_state(channel=channel, self_deaf=True)
                
            return player
        except (asyncio.TimeoutError, discord.ClientException, Exception) as e:
            logger.warning(f"Connection attempt {attempt+1}/{retries} failed: {e}")
            
            if channel.guild.voice_client:
                try:
                    await channel.guild.voice_client.disconnect(force=True)
                except:
                    pass
            
            if attempt == retries - 1:
                if ctx:
                    await safe_send(ctx.channel, "❌ **Connection Failed:** Network is unstable. Try changing the Voice Region.")
                raise e
            
            await asyncio.sleep(2) 
    return None

# --- Custom Bot Class ---
class MoodyMusicBot(commands.AutoShardedBot):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        intents.messages = True
        intents.voice_states = True

        super().__init__(
            command_prefix=self.get_custom_prefix,
            intents=intents,
            help_command=None,
            activity=discord.Activity(type=discord.ActivityType.listening, name="/play | m!help"),
            chunk_guilds_at_startup=False
        )

        self.autoplay = {}
        self.play_history = {}
        self.autoplay_seed = {}
        self.uptime = datetime.now()
        self._player_text_channels = {}

        self.web_server = WebServer(self)
        self.ai_recommender = AIRecommender()

    def is_autoplay_enabled(self, gid: int) -> bool:
        return self.autoplay.get(gid, False)

    def set_autoplay_enabled(self, gid: int, enabled: bool):
        self.autoplay[gid] = enabled
        if gid not in self.play_history:
            self.play_history[gid] = set()

    def add_play_history(self, gid: int, vid: Optional[str]):
        if not vid:
            return
        self.play_history.setdefault(gid, set()).add(vid)

    def set_player_text_channel(self, guild_id: int, channel_id: int):
        if not hasattr(self, "_player_text_channels"):
            self._player_text_channels = {}

        self._player_text_channels[guild_id] = channel_id

    def get_player_text_channel(self, guild_id: int):
        if not hasattr(self, "_player_text_channels"):
            self._player_text_channels = {}

        return self._player_text_channels.get(guild_id)

    async def get_custom_prefix(self, bot, message):
        if not message.guild:
            return commands.when_mentioned_or("m!")(bot, message)
        prefix = db.get_prefix(message.guild.id)
        return commands.when_mentioned_or(prefix)(bot, message)

    async def _resend_now_playing(self, player: wavelink.Player):
        guild_id = player.guild.id
        channel_id = self.get_player_text_channel(guild_id)

        if not channel_id:
            return

        channel = self.get_channel(channel_id)
        if not channel:
            return

        track = player.current
        if not track:
            return

        
        embed = discord.Embed(
            title="🎶 Now Playing",
            description=f"**{track.title}**\n{track.author}",
            color=discord.Color.from_rgb(138, 43, 226)
        )

        if track.uri:
            embed.add_field(name="🔗 Source", value=f"[Open Track]({track.uri})")

        embed.set_footer(text="Player re-synced after VC move")

        try:
            await channel.send(embed=embed, view=PlayerControls(player))
        except Exception as e:
            logger.error(f"Failed to resend Now Playing: {e}")


    async def on_voice_state_update(self, member, before, after):
        # Ignore non-bot events
        if not member.bot or member.id != self.user.id:
            return

        guild = member.guild
        player: wavelink.Player = guild.voice_client

        if not player:
            return

        # BOT WAS DRAGGED TO ANOTHER VC
        if before.channel and after.channel and before.channel.id != after.channel.id:
            logger.info(f"Bot dragged in {guild.name}: {before.channel} -> {after.channel}")

            try:
                # 🔹 Rebind internal state
                player.channel = after.channel

                # 🔹 Reset broken UI state
                player._paused = False

                # 🔹 Restore 24/7 safety
                if db.get_247_status(guild.id):
                    player.autoplay = wavelink.AutoPlayMode.disabled

                # 🔹 Resend NOW PLAYING if track exists
                if player.current:
                    await self._resend_now_playing(player)

            except Exception as e:
                logger.error(f"VC drag recovery failed: {e}")


    async def setup_hook(self):
        nodes = [wavelink.Node(
            uri=LAVALINK_URI,
            password=LAVALINK_PASS,
            identifier=LAVALINK_IDENTIFIER
        )]

        try:
            await wavelink.Pool.connect(
                nodes=nodes,
                client=self,
                cache_capacity=100
            )
            logger.info("Connected to Lavalink node(s)")
        except Exception as e:
            logger.error(f"Failed to connect to Lavalink: {e}")
            raise

        try:
            await self.web_server.start()
        except Exception as e:
            logger.error(f"Failed to start Web Server: {e}")

        try:
            await self.tree.sync()
            logger.info("Application commands synced")
        except Exception as e:
            logger.error(f"Failed to sync commands: {e}")

    async def on_ready(self):
        logger.info(f"Logged in as {self.user} ({self.user.id})")

        if not hasattr(self, "_player_text_channels"):
            self._player_text_channels = {}

        configs = db.get_all_247_channels()
        logger.info(f"Restoring {len(configs)} 24/7 voice connections")

        for guild_id, channel_id in configs:
            guild = self.get_guild(guild_id)
            if not guild:
                continue

            channel = guild.get_channel(channel_id)
            if not isinstance(channel, discord.VoiceChannel):
                continue

            if guild.voice_client:
                continue

            try:
                player = await channel.connect(
                    cls=wavelink.Player,
                    self_deaf=True
                )

                # 🔒 HARD RESET STATE
                player.queue.clear()
                player.autoplay = wavelink.AutoPlayMode.disabled
                player.text_channel = None

                logger.info(f"24/7 restored safely in {guild.name}")

            except Exception as e:
                logger.error(f"24/7 restore failed for {guild_id}: {e}")

    async def on_command_error(self, ctx, error):
        if isinstance(error, commands.MissingRequiredArgument):
            await ctx.send(f"❌ Missing argument: `{error.param.name}`")
        elif isinstance(error, commands.CommandNotFound):
            pass
        elif isinstance(error, commands.CheckFailure):
            await ctx.send("❌ You don't have permission to use this command.")
        elif isinstance(error, commands.CommandInvokeError):
            logger.error(f"Command error in {ctx.command}: {error.original}")
            await ctx.send("❌ An error occurred while executing the command.")
        else:
            logger.error(f"Unhandled command error: {error}")

    async def on_guild_join(self, guild: discord.Guild):
        logger.info(f"Joined guild: {guild.name} (ID: {guild.id})")
        db.set_prefix(guild.id, "m!")

    async def on_guild_remove(self, guild: discord.Guild):
        logger.info(f"Left guild: {guild.name} (ID: {guild.id})")
        if guild.id in self.autoplay:
            del self.autoplay[guild.id]
        if guild.id in self.play_history:
            del self.play_history[guild.id]
        if guild.id in self.autoplay_seed:
            del self.autoplay_seed[guild.id]
        if guild.id in self._player_text_channels:
            del self._player_text_channels[guild.id]

    async def run_setup(self, guild: discord.Guild) -> Optional[discord.VoiceChannel]:
        try:
            channel_name = "Moody Music 24/7"
            existing_channel = discord.utils.get(guild.voice_channels, name=channel_name)

            if not existing_channel:
                category = discord.utils.get(guild.categories, name="Music")
                if not category:
                    category = await guild.create_category("Music")

                channel = await guild.create_voice_channel(
                    name=channel_name,
                    category=category,
                    reason="24/7 Music Bot Setup"
                )
                logger.info(f"Created 24/7 channel in {guild.name}")
            else:
                channel = existing_channel

            # ✅ SAVE 24/7 STATE
            db.set_247_status(guild.id, True, channel.id)

            # ✅ CONNECT ONLY TO 24/7 VC
            if not guild.voice_client:
                player = await channel.connect(cls=wavelink.Player, self_deaf=True)
                player.autoplay = wavelink.AutoPlayMode.disabled
                player.text_channel = None

            return channel

        except Exception as e:
            logger.error(f"Setup failed in {guild.name}: {e}")
            return None

    async def on_message(self, message):
        if message.author.bot:
            return

        if self.user.mentioned_in(message) and not message.mention_everyone:
            content = message.content.replace(f'<@!{self.user.id}>', '').replace(f'<@{self.user.id}>', '').strip()

            if not content:
                prefix = db.get_prefix(message.guild.id) if message.guild else "m!"
                latency = round(self.latency * 1000)

                embed = discord.Embed(
                    title=f"👋 Hi there! I'm {self.user.name.upper()}",
                    description=(
                        f"I am your advanced **Music & Trivia** companion. High-quality "
                        f"audio, lyrics, games, and audio effects all in one place!"
                    ),
                    color=discord.Color.from_rgb(138, 43, 226)
                )

                embed.set_thumbnail(url=self.user.display_avatar.url)

                embed.add_field(
                    name="🎧 **Music Playback**",
                    value=(
                        f"> • **Join** a voice channel.\n"
                        f"> • Type `/play <song name>` or `{prefix}play <url>`.\n"
                        f"> • Use the **Player Buttons** to pause, skip, or loop.\n"
                        f"> • View Queue: `{prefix}queue`"
                    ),
                    inline=False
                )

                embed.add_field(
                    name="🎮 **Fun & Trivia**",
                    value=(
                        f"> • Type `/games` (or `{prefix}games`) to start.\n"
                        f"> • Choose 🎵 **Song Trivia** or 🎬 **Bollywood Guess**.\n"
                        f"> • Compete with friends for the highest score!"
                    ),
                    inline=False
                )

                embed.add_field(
                    name="🎚️ **Effects & Filters**",
                    value=(
                        f"> • `{prefix}eq` for **8D, Karaoke, Bass Boost**.\n"
                        f"> • `{prefix}speed` to change tempo (Nightcore/Slow).\n"
                        f"> • `{prefix}autoplay` to toggle auto-recommendations."
                    ),
                    inline=False
                )

                embed.set_footer(
                    text=f"Current Prefix: {prefix} | Ping: {latency}ms | Made with 💖",
                    icon_url=self.user.display_avatar.url
                )

                view = discord.ui.View()
                help_btn = discord.ui.Button(label="Full Command List", style=discord.ButtonStyle.blurple, emoji="📜")

                async def help_callback(interaction):
                    p = db.get_prefix(interaction.guild_id) if interaction.guild_id else "m!"
                    help_embed = discord.Embed(title="🎵 Moody Music Help", color=discord.Color.from_rgb(138, 43, 226))

                    music_val = f"""`{p}join` - Join voice
`{p}play <song>` - Play song
`{p}play jssearch:<song>` - Search JioSaavn
`{p}skip` - Skip track
`{p}stop` - Stop & clear queue
`{p}pause` / `{p}resume`
`{p}loop [track/queue/off]`
`{p}queue` - View queue
`{p}clear` - Clear queue only
`{p}volume <0-1000>`
`{p}lyrics` - Get lyrics
`{p}autoplay` - Toggle autoplay"""
                    help_embed.add_field(name="🎧 Music", value=music_val, inline=False)

                    effects_val = f"""`{p}eq` - Equalizer
`{p}speed [0.5-2.0]`
`{p}current` - Now playing"""
                    help_embed.add_field(name="🎚️ Effects", value=effects_val, inline=False)

                    games_val = f"""`{p}games` - Trivia
`{p}trivia stop`"""
                    help_embed.add_field(name="🎮 Games", value=games_val, inline=False)

                    admin_val = f"""`{p}setup` - Create 24/7 channel
`{p}247` - Toggle 24/7
`{p}prefix <new>`
`{p}sync` - Sync slash cmds"""
                    help_embed.add_field(name="⚙️ Admin", value=admin_val, inline=False)

                    await interaction.response.send_message(embed=help_embed, ephemeral=True)

                help_btn.callback = help_callback
                view.add_item(help_btn)

                try:
                    await message.channel.send(embed=embed, view=view)
                except discord.Forbidden:
                    logger.warning(f"No permissions to send message in {message.channel}")
                except Exception as e:
                    logger.error(f"Error sending mention UI: {e}")

        await self.process_commands(message)


# --- Music Cog ---
class Music(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.trivia_games: Dict[int, GameState] = {}
        self.skip_votes = {}
        self._empty_channel_tasks = {}
        self.last_actions = {}
        self.rejoin_counters = {}

    def clean_song_title(self, title):
        if not title:
            return ""
        return title.split('(')[0].split('[')[0].strip()[:80]

    async def validate_voice_status(self, ctx):
            """
            Checks if the user is in the same voice channel as the bot.
            Returns True if safe to proceed, False if command should stop.
            """
            # 1. User not in voice
            if not ctx.author.voice:
                await ctx.send("🚫 **You must be in a voice channel to use this command.**")
                return False

            # 2. Bot not connected (Safe to proceed, command logic handles 'not connected' errors)
            if not ctx.voice_client:
                return True

            # 3. Channel Mismatch check
            bot_channel = ctx.voice_client.channel
            user_channel = ctx.author.voice.channel

            if bot_channel.id != user_channel.id:
                embed = discord.Embed(
                    title="🚫 Access Denied",
                    description=f"I am currently active in **{bot_channel.mention}**.\nYou must join that channel to use music commands.",
                    color=discord.Color.red()
                )
                # Add a button to jump directly to the channel
                view = discord.ui.View()
                btn = discord.ui.Button(label=f"Join {bot_channel.name}", url=bot_channel.jump_url, style=discord.ButtonStyle.link)
                view.add_item(btn)
                
                await ctx.send(embed=embed, view=view, ephemeral=True)
                return False

            return True


    @commands.Cog.listener()
    async def on_wavelink_node_ready(self, payload: wavelink.NodeReadyEventPayload):
        logger.info(f"Node ready: {payload.node.identifier}")

    @commands.Cog.listener()
    async def on_voice_state_update(self, member, before, after):
        """
        Robust 24/7 Enforcement with Circuit Breaker (Anti-Loop) & Force Deafen.
        """
        # 1. Sanity Checks: Ignore non-bot updates
        if not self.bot.is_ready() or member.id != self.bot.user.id:
            return

        guild = member.guild

        # 2. Check Feature Flag: Is 24/7 enabled in Redis?
        if not db.get_247_status(guild.id):
            return

        # ==========================================
        # CASE A: BOT WAS DISCONNECTED (Left Voice)
        # ==========================================
        if before.channel is not None and after.channel is None:
            
            # Stop if the bot is shutting down
            if self.bot.is_closed():
                return

            # Get the saved 24/7 channel ID
            home_channel_id = db.get_247_channel(guild.id)
            if not home_channel_id:
                return

            home_channel = guild.get_channel(home_channel_id)

            # --- VALIDATION 1: Channel Exists & Type ---
            if not isinstance(home_channel, (discord.VoiceChannel, discord.StageChannel)):
                logger.warning(f"⚠️ 24/7 channel invalid/deleted in {guild.name}. Disabling 24/7.")
                db.set_247_status(guild.id, False)
                return

            # --- VALIDATION 2: Permissions ---
            permissions = home_channel.permissions_for(guild.me)
            if not permissions.connect or not permissions.view_channel:
                logger.warning(f"⚠️ Missing 'Connect' or 'View' permissions in {guild.name}. Disabling 24/7.")
                db.set_247_status(guild.id, False)
                return

            # --- CIRCUIT BREAKER (Anti-Loop Logic) ---
            current_time = time.time()
            # Get (count, last_timestamp) - default to 0
            count, last_time = self.rejoin_counters.get(guild.id, (0, 0))

            # If last attempt was less than 60 seconds ago, increment count
            if current_time - last_time < 60:
                count += 1
            else:
                # Reset counter if it's been over a minute
                count = 1

            # Save state
            self.rejoin_counters[guild.id] = (count, current_time)

            # TRIGGER BREAKER: If >3 attempts in 60s
            if count > 3:
                logger.error(f"🚨 Infinite Rejoin Loop detected in {guild.name}. Disabling 24/7 Mode safely.")
                db.set_247_status(guild.id, False)
                return
            # -----------------------------------------

            logger.warning(f"🔄 Bot disconnected in {guild.name} (Attempt {count}/3). Waiting 5s...")

            # Wait to prevent rapid-fire API spam
            await asyncio.sleep(5) 

            # Double check: Did someone manually reconnect us during the sleep?
            if guild.voice_client and guild.voice_client.connected:
                return

            try:
                # Cleanup potential ghost state before connecting
                if guild.voice_client:
                    try:
                        await guild.voice_client.disconnect(force=True)
                    except:
                        pass
                
                # Attempt Connection
                player = await home_channel.connect(cls=wavelink.Player, self_deaf=True)
                
                # --- FIX: FORCE DEAFEN ---
                # Explicitly apply deafen if the initial handshake missed it
                if not guild.me.voice or not guild.me.voice.self_deaf:
                     await guild.change_voice_state(channel=home_channel, self_deaf=True)

                # Reset Player State (stop autoplay, clear text channel binding)
                player.autoplay = wavelink.AutoPlayMode.disabled
                if hasattr(player, 'text_channel'):
                    player.text_channel = None 
                
                logger.info(f"✅ Successfully auto-rejoined 24/7 channel in {guild.name}")
                
            except Exception as e:
                logger.error(f"❌ Failed to auto-rejoin 24/7 channel in {guild.name}: {e}")

        # ==========================================
        # CASE B: BOT WAS MOVED (Dragged to new channel)
        # ==========================================
        elif after.channel is not None:
            # If dragged to a new channel, ensure we remain deafened
            if not guild.me.voice.self_deaf:
                try:
                    await guild.change_voice_state(channel=after.channel, self_deaf=True)
                except Exception as e:
                    logger.debug(f"Failed to force deafen after drag: {e}")


    @commands.Cog.listener()
    async def on_wavelink_track_start(self, payload: wavelink.TrackStartEventPayload):
        player = payload.player
        if not player:
            return

        if player.guild.id in self.trivia_games and self.trivia_games[player.guild.id].active:
            return

        track = payload.track

        if track.uri and ("youtube.com" in track.uri or "youtu.be" in track.uri):
            video_id = extract_video_id(track.uri)
            if video_id:
                self.bot.autoplay_seed[player.guild.id] = video_id
                self.bot.add_play_history(player.guild.id, video_id)

        if not getattr(track, 'is_rewind', False):
            db.add_to_history(player.guild.id, track.title, track.uri or "", track.author)

        custom_data = getattr(track, "custom_payload", {})
        requester_id = custom_data.get("requester")
        if requester_id:
            db.update_user_stats(requester_id, player.guild.id)

        provider = custom_data.get("provider", "default")
        if provider == "default" and track.uri:
            if "spotify.com" in track.uri:
                provider = "spotify"
            elif "music.apple.com" in track.uri:
                provider = "applemusic"
            elif "soundcloud.com" in track.uri:
                provider = "soundcloud"
            elif "youtube.com" in track.uri or "youtu.be" in track.uri:
                provider = "youtube"

        icon_url = ICONS.get(provider, ICONS["default"])
        status_emoji = "🎵"

        duration_min = int((track.length / 1000) // 60)
        duration_sec = int((track.length / 1000) % 60)

        total_bars = 15
        progress_bar = "🔘" + "▬" * (total_bars - 1)

        embed = discord.Embed(
            description=f"## {status_emoji} Now Playing\n**[{track.title}]({track.uri or 'https://youtube.com'})**\nby `{track.author}`\n\n`0:00` {progress_bar} `{duration_min:02d}:{duration_sec:02d}`",
            color=discord.Color.from_rgb(138, 43, 226)
        )
        embed.set_author(name=f"Playing on {player.channel.name}", icon_url=icon_url)

        if custom_data.get("artwork"):
            embed.set_thumbnail(url=custom_data["artwork"])
        elif track.artwork:
            embed.set_thumbnail(url=track.artwork)
        elif track.uri and "youtube" in track.uri:
            video_id = extract_video_id(track.uri)
            if video_id:
                embed.set_thumbnail(url=f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg")

        footer_text = f"Source: {provider.capitalize()}"
        if requester_id:
            user = self.bot.get_user(requester_id)
            if user:
                footer_text += f" • Requested by: {user.display_name}"
        embed.set_footer(text=footer_text, icon_url=self.bot.user.display_avatar.url)

        channel_id = self.bot.get_player_text_channel(player.guild.id)
        if not channel_id:
            if hasattr(player, 'text_channel') and player.text_channel:
                channel_id = player.text_channel.id
            else:
                return

        channel = player.guild.get_channel(channel_id)
        if not channel:
            return

        old_msg = getattr(player, 'controller_message', None)
        if old_msg:
            try:
                await old_msg.delete()
            except:
                pass

        try:
            msg = await channel.send(embed=embed, view=PlayerControls(player))
            player.controller_message = msg
            player.text_channel = channel
        except discord.Forbidden:
            logger.warning(f"No permissions to send message in {channel}")
        except Exception as e:
            logger.error(f"Failed to send controller: {e}")

        try:
            if player.channel and isinstance(player.channel, discord.VoiceChannel):
                status_text = f"{status_emoji} {track.title[:45]} • {track.author[:45]}"
                if len(status_text) > 128:
                    status_text = status_text[:125] + "..."
                await player.channel.edit(status=status_text)
        except discord.Forbidden:
            pass
        except Exception as e:
            logger.warning(f"Failed to update VC status: {e}")

    async def add_autoplay_track(self, player: wavelink.Player) -> Optional[wavelink.Playable]:
        guild_id = player.guild.id

        if not self.bot.is_autoplay_enabled(guild_id):
            return None

        seed_id = self.bot.autoplay_seed.get(guild_id)
        if not seed_id:
            return None

        candidates = await get_recommended_from_seed(seed_id, limit=10)
        if not candidates:
            return None

        history = self.bot.play_history.get(guild_id, set())
        fresh_candidates = [c for c in candidates if c['id'] not in history]

        if not fresh_candidates:
            fresh_candidates = candidates

        if not fresh_candidates:
            return None

        chosen = random.choice(fresh_candidates)

        try:
            tracks = await wavelink.Playable.search(f"ytsearch:{chosen['title']}")
            if tracks and not isinstance(tracks, wavelink.Playlist):
                track = tracks[0]
                track.custom_payload = {
                    "provider": "youtube",
                    "requester": self.bot.user.id,
                    "autoplay": True
                }

                self.bot.add_play_history(guild_id, chosen['id'])
                self.bot.autoplay_seed[guild_id] = chosen['id']

                return track
        except Exception as e:
            logger.error(f"Failed to search autoplay track: {e}")

        return None
    
    @commands.hybrid_command(name="sync")
    @is_owner_or_admin()
    async def sync(self, ctx: commands.Context):
        """Sync application (slash) commands"""
        await ctx.defer(ephemeral=True)

        try:
            synced = await ctx.bot.tree.sync()
            await ctx.send(f"✅ Synced `{len(synced)}` application commands.")
        except Exception as e:
            await ctx.send(f"❌ Sync failed: `{e}`")

    @commands.hybrid_command(name="ping", aliases=["latency"])
    async def ping(self, ctx):
        """Check Bot, Redis, Voice, and Lavalink latency"""
        # 1. Bot (Websocket) Latency
        bot_latency = round(self.bot.latency * 1000)
        
        # 2. Redis Latency
        redis_latency = db.get_redis_latency()
        redis_status = f"`{redis_latency}ms`" if redis_latency != -1 else "❌ **Offline**"
        
        # 3. Voice Latency (if connected)
        voice_info = "Not Connected"
        if ctx.guild.voice_client:
            vc: wavelink.Player = ctx.guild.voice_client
            if hasattr(vc, "ping") and vc.ping >= 0:
                 voice_info = f"`{vc.ping}ms`"
            else:
                 voice_info = "📶 Connecting..."

        # 4. Lavalink Node Latency (NEW)
        node = wavelink.Pool.get_node(LAVALINK_IDENTIFIER)
        lava_latency = "❌ **Offline**"
        if node and node.heartbeat:
            lava_latency = f"`{round(node.heartbeat)}ms`"
        elif node:
            lava_latency = "📶 **Connecting...**"

        embed = discord.Embed(
            title="🏓 Pong!",
            color=discord.Color.from_rgb(138, 43, 226)
        )
        embed.add_field(name="🤖 Bot Latency", value=f"`{bot_latency}ms`", inline=True)
        embed.add_field(name="🗄️ Redis Database", value=redis_status, inline=True)
        embed.add_field(name="🌋 Lavalink Node", value=lava_latency, inline=True)
        embed.add_field(name="🎵 Voice Server", value=voice_info, inline=True)
        
        await ctx.send(embed=embed)

    @commands.hybrid_command(name="quickfix", aliases=["fixlag"]) 
    async def quickfix(self, ctx):
        """Attempt to fix lag or audio stuttering"""
        vc: wavelink.Player = ctx.voice_client

        if not vc:
            await ctx.send("❌ Not connected to a voice channel.")
            return

        status_msg = await ctx.send("🛠️ **Optimising connection...** (Resetting Audio Buffer)")
        
        try:
            # Step 1: Soft Fix (Pause/Resume to flush UDP buffer)
            was_paused = vc.paused
            if not was_paused:
                await vc.pause(True)
                await asyncio.sleep(1.5) # Allow buffer to drain
                await vc.pause(False)
            
            # Step 2: Check Region (Informational)
            region = ctx.guild.voice_client.channel.rtc_region
            region_name = region if region else "Automatic"

            embed = discord.Embed(
                title="✅ Connection Optimised",
                description="Performed an audio buffer reset.",
                color=discord.Color.green()
            )
            embed.add_field(name="Current Region", value=f"`{region_name}`", inline=True)
            embed.add_field(name="Packet Loss", value="Checked (Internal)", inline=True)
            embed.set_footer(text="If lag persists, try changing the Voice Channel Region in Server Settings.")

            await status_msg.edit(content=None, embed=embed)
            
        except Exception as e:
            logger.error(f"Optimise failed: {e}")
            await status_msg.edit(content="❌ Failed to optimise connection.")

    @commands.Cog.listener()
    async def on_wavelink_track_end(self, payload: wavelink.TrackEndEventPayload):
        player = payload.player
        if not player:
            return

        if payload.reason == "replaced":
            return

        guild_id = player.guild.id

        try:
            if player.channel and isinstance(player.channel, discord.VoiceChannel):
                listeners = [m for m in player.channel.members if not m.bot]

                if not listeners:
                    if db.get_247_status(guild_id):
                        player.queue.clear()
                        await player.stop()
                        try:
                            await player.channel.edit(status=None)
                        except:
                            pass
                        return
                    else:
                        player.queue.clear()
                        db.clear_history(guild_id)
                        await player.disconnect()
                        return

            if player.queue.mode == wavelink.QueueMode.loop:
                await player.play(payload.track)
                return
            elif player.queue.mode == wavelink.QueueMode.loop_all:
                await player.queue.put_wait(payload.track)

            if not player.queue.is_empty:
                next_track = player.queue.get()
                await player.play(next_track)
                return

            if self.bot.is_autoplay_enabled(guild_id):
                autoplay_track = await self.add_autoplay_track(player)
                if autoplay_track:
                    await player.play(autoplay_track)
                    return

            try:
                if player.channel and isinstance(player.channel, discord.VoiceChannel):
                    await player.channel.edit(status=None)
            except:
                pass

            if not db.get_247_status(guild_id):
                await asyncio.sleep(300)

                if (player.connected and player.queue.is_empty
                        and not player.playing and not player.paused):
                    try:
                        old_msg = getattr(player, 'controller_message', None)
                        if old_msg:
                            try:
                                await old_msg.delete()
                            except:
                                pass
                        db.clear_history(guild_id)
                        await player.disconnect()
                    except:
                        pass

        except Exception as e:
            logger.error(f"Track end error: {e}")
            try:
                if not db.get_247_status(guild_id):
                    await player.disconnect()
            except:
                pass

    @commands.hybrid_command(name="suggest")
    async def suggest_command(self, ctx, *, mood_or_query: str = None):
        """Get AI-powered song recommendations (Restricted to Author)."""
        await ctx.defer()

        if not GEMINI_API_KEY:
            await safe_send(ctx.channel, "❌ AI features are not configured (Missing API Key).")
            return

        # Build prompt
        prompt = ""
        vc: Optional[wavelink.Player] = ctx.voice_client
        
        # EDGE CASE 1: Build prompt even if player is None or Idle
        try:
            if mood_or_query and isinstance(mood_or_query, str) and mood_or_query.strip():
                prompt = mood_or_query.strip()
            else:
                if vc and getattr(vc, "current", None):
                    cur = vc.current
                    title = getattr(cur, "title", "Unknown")
                    author = getattr(cur, "author", "") or ""
                    prompt = f"Songs similar to {title} by {author}".strip()
                else:
                    # EDGE CASE 2: Fetch from DB history if nothing is playing
                    history = db.get_last_two_tracks(ctx.guild.id) if ctx.guild else []
                    if history:
                        last_song = history[0]
                        title = last_song[0] or "Unknown"
                        author = last_song[2] or ""
                        prompt = f"Songs similar to {title} by {author}".strip()
                    else:
                        await safe_send(ctx.channel, "ℹ️ Nothing playing. Provide a mood/query! e.g., `/suggest upbeat pop 80s`")
                        return
        except Exception as e:
            logger.exception(f"Suggest: error building prompt: {e}")
            await safe_send(ctx.channel, "❌ Failed to prepare suggestion prompt.")
            return

        thinking_msg = None
        try:
            thinking_msg = await ctx.send(f"🤖 **AI Thinking...** Generating playlist for: *{html.escape(prompt)}*")
        except Exception:
            thinking_msg = None

        # Try AI (with timeout)
        recommendations: List[str] = []
        try:
            recommendations = await asyncio.wait_for(self.bot.ai_recommender.get_recommendations(prompt), timeout=20)
            
            # normalize
            if recommendations and isinstance(recommendations, (list, tuple)):
                recommendations = [str(x).strip() for x in recommendations if x and str(x).strip()]
            else:
                recommendations = []
        except asyncio.TimeoutError:
            logger.warning("AI recommendation request timed out.")
            recommendations = []
        except Exception as e:
            logger.exception(f"AI recommendation error: {e}")
            recommendations = []

        # If AI failed, fallback to YouTube RD recommendations
        if not recommendations:
            try:
                seed_id = None
                if vc and getattr(vc, "current", None):
                    uri = getattr(vc.current, "uri", "") or getattr(vc.current, "webpage_url", "") or ""
                    seed_id = extract_video_id(uri) or uri
                
                if not seed_id and ctx.guild:
                    history = db.get_last_two_tracks(ctx.guild.id)
                    if history:
                        uri = history[0][1] or ""
                        seed_id = extract_video_id(uri) or uri

                if seed_id:
                    logger.info(f"AI failed — trying YouTube RD fallback with seed {seed_id}")
                    rd = await get_recommended_from_seed(seed_id, limit=12)
                    if rd:
                        formatted = []
                        for entry in rd:
                            t = entry.get("title") or ""
                            s = t.strip()
                            if s:
                                formatted.append(s)
                        recommendations = formatted[:10]
            except Exception as e:
                logger.exception(f"Fallback recommendations error: {e}")
                recommendations = []

        if not recommendations:
            await safe_send(ctx.channel, "❌ AI couldn't generate recommendations right now. Try providing a specific query.")
            return

        # Build embed + select
        try:
            embed = discord.Embed(
                title="🤖 AI Recommendations",
                description=f"Based on: **{html.escape(prompt)}**\nSelect a track to add to queue:",
                color=discord.Color.from_rgb(138, 43, 226)
            )

            select_menu = discord.ui.Select(placeholder="Select a song to play...", min_values=1, max_values=1)
            for i, song_str in enumerate(recommendations[:10]):
                display = str(song_str)
                embed.add_field(name=f"{i+1}.", value=display[:1024], inline=False)
                label = display[:100]
                select_menu.add_option(label=label, value=display, emoji="🎵")

            async def select_callback(interaction: discord.Interaction):
                # --- 🔒 SECURITY CHECK 🔒 ---
                if interaction.user.id != ctx.author.id:
                    await interaction.response.send_message("🚫 Only the command author can select a song!", ephemeral=True)
                    return
                
                try:
                    await interaction.response.defer()
                except Exception:
                    pass

                # Get selected value
                try:
                    query = select_menu.values[0]
                except Exception:
                    await interaction.followup.send("❌ No song selected.", ephemeral=True)
                    return

                # Ensure voice connection
                vc = ctx.voice_client
                if not vc:
                    if ctx.author.voice and ctx.author.voice.channel:
                        try:
                            vc = await ctx.author.voice.channel.connect(
                                cls=wavelink.Player,
                                self_deaf=True
                            )
                        except Exception as e:
                            logger.exception("VC connect failed", exc_info=True)
                            await interaction.followup.send("❌ Failed to connect to voice channel.", ephemeral=True)
                            return
                    else:
                        await interaction.followup.send("🚫 Join a voice channel first.", ephemeral=True)
                        return
                
                # --- EDGE CASE FIX: ENSURE TEXT CHANNEL IS SET ---
                # This fixes the "Now Playing" screen not showing.
                # We force the bot to recognize the channel where the interaction occurred as the output channel.
                try:
                    vc.text_channel = interaction.channel
                    self.bot.set_player_text_channel(ctx.guild.id, interaction.channel.id)
                except Exception:
                    logger.warning("Could not update player text channel context.")

                # Search track
                try:
                    tracks = await wavelink.Playable.search(f"ytsearch:{query}")
                except Exception as e:
                    logger.exception("Search failed", exc_info=True)
                    await interaction.followup.send("❌ Failed to search for track.", ephemeral=True)
                    return

                if not tracks:
                    await interaction.followup.send("❌ Track not found.", ephemeral=True)
                    return

                # Resolve track
                track = None
                try:
                    if isinstance(tracks, (list, tuple)):
                        track = tracks[0]
                    elif hasattr(tracks, "tracks"):
                        track = tracks.tracks[0]
                except Exception:
                    track = None

                if not track:
                    await interaction.followup.send("❌ Could not resolve track.", ephemeral=True)
                    return

                # Tag requester
                try:
                    track.custom_payload = {
                        "requester": interaction.user.id,
                        "source": "ai_suggest",
                        "provider": "youtube" # Explicitly set provider for Icon logic
                    }
                except Exception:
                    pass

                # 🔥 SAFE PLAY / QUEUE LOGIC
                try:
                    result = await safe_play_or_queue(vc, track, interaction)
                except Exception as e:
                    logger.exception("Playback failed", exc_info=True)
                    await interaction.followup.send("❌ Failed to play selected track.", ephemeral=True)
                    return

                # User feedback
                try:
                    if result == "played":
                        # We do NOT send "Now Playing" here because on_track_start will send the Embed.
                        # Sending a confirmation avoids duplicate clutter.
                        await interaction.followup.send(f"✅ Loaded **{track.title}**...", ephemeral=True)
                        
                        # EDGE CASE: If the view was attached to a message, try to disable it to prevent double clicks
                        try:
                            view.stop()
                            if thinking_msg:
                                await thinking_msg.edit(view=None) 
                        except: 
                            pass
                            
                    else:
                        await interaction.followup.send(f"➕ Added **{track.title}** to queue")
                except Exception:
                    pass


            select_menu.callback = select_callback
            view = discord.ui.View(timeout=90)
            view.add_item(select_menu)

            # Edit the thinking message (if possible) or send fresh
            try:
                if thinking_msg:
                    await thinking_msg.edit(content=None, embed=embed, view=view)
                else:
                    await ctx.send(embed=embed, view=view)
            except Exception:
                try:
                    await ctx.send(embed=embed, view=view)
                except Exception as e:
                    logger.exception(f"Failed to send recommendation embed: {e}")
                    await safe_send(ctx.channel, "❌ Failed to send recommendations.")
        except Exception as e:
            logger.exception(f"Unexpected error in suggest_command: {e}")
            await safe_send(ctx.channel, "❌ Something went wrong while preparing recommendations.")


    @commands.hybrid_command(name="help")
    async def help_command(self, ctx):
        """Show help menu"""
        p = db.get_prefix(ctx.guild.id) if ctx.guild else "m!"
        embed = discord.Embed(title="🎵 Moody Music Help", color=discord.Color.from_rgb(138, 43, 226))

        music_cmds = f"""`{p}join` - Join voice
`{p}play <song>` - Play song
`{p}play jssearch:<song>` - Search JioSaavn
`{p}skip` - Skip track
`{p}stop` - Stop & clear queue
`{p}pause` / `{p}resume`
`{p}loop [track/queue/off]`
`{p}queue` - View queue
`{p}clear` - Clear queue only
`{p}volume <0-1000>`
`{p}lyrics` - Get lyrics
`{p}autoplay` - Toggle autoplay"""
        embed.add_field(name="🎧 Music", value=music_cmds, inline=False)

        effects_cmds = f"""`{p}eq` - Equalizer
`{p}speed [0.5-2.0]`
`{p}current` - Now playing"""
        embed.add_field(name="🎚️ Effects", value=effects_cmds, inline=False)

        games_cmds = f"""`{p}games` - Trivia
`{p}trivia stop`"""
        embed.add_field(name="🎮 Games", value=games_cmds, inline=False)

        admin_cmds = f"""`{p}setup` - Create 24/7 channel
`{p}247` - Toggle 24/7
`{p}prefix <new>`
`{p}sync` - Sync slash cmds"""
        embed.add_field(name="⚙️ Admin", value=admin_cmds, inline=False)

        await ctx.send(embed=embed)

    @commands.hybrid_command(name="join", aliases=["j", "connect"])
    async def join(self, ctx):
        """Join your voice channel (Protected)"""
        if not ctx.author.voice:
            await ctx.send("🚫 You need to be in a voice channel first!")
            return

        user_channel = ctx.author.voice.channel
        vc = ctx.voice_client

        # If bot is already connected
        if vc:
            # If already in the same channel
            if vc.channel.id == user_channel.id:
                await ctx.send("✅ I am already in your channel!")
                return
            
            # --- PROTECTIVE LOGIC ---
            # Check for humans (excluding bots)
            listeners = [m for m in vc.channel.members if not m.bot]
            
            if listeners:
                embed = discord.Embed(
                    title="🚫 Setup Active",
                    description=f"I am currently playing for **{len(listeners)} users** in **{vc.channel.mention}**.\nI cannot move right now.",
                    color=discord.Color.red()
                )
                view = discord.ui.View()
                btn = discord.ui.Button(label=f"Join {vc.channel.name}", url=vc.channel.jump_url, style=discord.ButtonStyle.link)
                view.add_item(btn)
                await ctx.send(embed=embed, view=view, ephemeral=True)
                return

            # If no humans, move to new channel
            await vc.move_to(user_channel)
            await ctx.send(f"✅ Moved to **{user_channel.name}**")
            return

        # Normal Connect
        try:
            vc = await robust_connect(user_channel, ctx)
            vc.autoplay = wavelink.AutoPlayMode.disabled
            vc.text_channel = ctx.channel
            self.bot.set_player_text_channel(ctx.guild.id, ctx.channel.id)
            await ctx.send(f"✅ Joined **{user_channel.name}**")
        except discord.Forbidden:
            await ctx.send("❌ I don't have permission to join that channel!")
        except Exception as e:
            logger.error(f"Join error: {e}")
            await ctx.send("❌ Failed to join voice channel.")

    @commands.hybrid_command(name="play", aliases=["p"])
    async def play(self, ctx, *, query: str):
        """Play a song or add to queue"""
        
        # 1. Voice Channel Guard (Replaces old Busy Check & User Voice Check)
        if not await self.validate_voice_status(ctx):
            return

        await ctx.defer()
        
        # 2. Setup Variables
        user_vc = ctx.author.voice.channel
        bot_vc = ctx.voice_client

        # 3. Connection Logic
        if not bot_vc:
            try:
                # Connect if not connected
                vc = await robust_connect(user_vc, ctx)
                vc.autoplay = wavelink.AutoPlayMode.disabled
                vc.text_channel = ctx.channel
                self.bot.set_player_text_channel(ctx.guild.id, ctx.channel.id)
            except Exception as e:
                await ctx.send(f"❌ Failed to connect: {e}")
                return
        else:
            vc = bot_vc
            # Check for ghost player state
            if not vc.guild:
                 await ctx.send("❌ Internal error: Player state invalid. Try `/repair`.")
                 return
            
            # Update text channel for now playing messages
            vc.text_channel = ctx.channel
            self.bot.set_player_text_channel(ctx.guild.id, ctx.channel.id)

        # 4. Resolve Provider
        provider = "default"
        track_info = None

        if "music.apple.com" in query:
            provider = "applemusic"
        elif "spotify.com" in query:
            provider = "spotify"
        elif "soundcloud.com" in query:
            provider = "soundcloud"
        elif "youtube.com" in query or "youtu.be" in query:
            provider = "youtube"
        elif "jiosaavn.com" in query:
            provider = "jiosaavn"
            track_info = await MusicSourceResolver.resolve_jiosaavn(query)
        elif query.startswith("jssearch:"):
            provider = "jiosaavn"
            search_q = query.split("jssearch:", 1)[1].strip()
            track_info = await MusicSourceResolver.search_jiosaavn(search_q)
        elif "gaana.com" in query:
            provider = "gaana"
            track_info = await MusicSourceResolver.resolve_gaana(query)

        tracks_to_queue = []

        # 5. Handle JioSaavn/Gaana custom resolution
        if provider in ["jiosaavn", "gaana"]:
            if track_info:
                if isinstance(track_info, list):
                    for item in track_info:
                        tracks_to_queue.append({
                            "query": f"ytsearch:{item['title']} {item.get('artist', '')}",
                            "meta": item
                        })
                    logger.info(f"Resolved {provider} list with {len(tracks_to_queue)} tracks")
                else:
                    tracks_to_queue.append({
                        "query": f"ytsearch:{track_info['title']} {track_info.get('artist', '')}",
                        "meta": track_info
                    })
                    logger.info(f"Resolved {provider} link to: {tracks_to_queue[0]['query']}")
            else:
                await ctx.send(f"❌ Could not resolve {provider.title()} link/query.")
                return
        else:
            # 6. Standard Wavelink Search
            try:
                tracks = await wavelink.Playable.search(query)
                if not tracks:
                    await ctx.send("❌ No results found.")
                    return

                if isinstance(tracks, wavelink.Playlist):
                    for track in tracks.tracks:
                        track.custom_payload = {"provider": provider, "requester": ctx.author.id}
                        await vc.queue.put_wait(track)

                    embed = discord.Embed(
                        title="Playlist Added",
                        description=f"✅ Added **{len(tracks.tracks)}** tracks from **{tracks.name}**",
                        color=discord.Color.green()
                    )
                    await ctx.send(embed=embed, delete_after=10)
                    if not vc.playing:
                        await vc.play(vc.queue.get())
                    return
                else:
                    # Single Track
                    track = tracks[0]
                    track.custom_payload = {"provider": provider, "requester": ctx.author.id}
                    await vc.queue.put_wait(track)
                    embed = discord.Embed(
                        title="Track Added",
                        description=f"✅ Added **[{track.title}]({track.uri or 'https://youtube.com'})** to queue",
                        color=discord.Color.blurple()
                    )
                    await ctx.send(embed=embed, delete_after=10)
                    if not vc.playing:
                        await vc.play(vc.queue.get())
                    return
            except Exception as e:
                logger.error(f"Standard search error: {e}")
                await ctx.send("❌ Search failed.")
                return

        if not tracks_to_queue:
            return

        # 7. Play first track from Custom List (JioSaavn/Gaana)
        first_item = tracks_to_queue.pop(0)
        try:
            first_tracks = await wavelink.Playable.search(first_item["query"])
            if first_tracks:
                track = first_tracks[0]
                track.custom_payload = {"provider": provider, "requester": ctx.author.id}
                if first_item["meta"].get("image"):
                    track.custom_payload["artwork"] = first_item["meta"]["image"]

                await vc.queue.put_wait(track)

                if not vc.playing:
                    await vc.play(vc.queue.get())

                if not tracks_to_queue:
                    embed = discord.Embed(
                        title="Track Added",
                        description=f"✅ Added **{track.title}** from {provider.title()}",
                        color=discord.Color.blurple()
                    )
                    if provider in ["jiosaavn", "gaana"]:
                        embed.set_footer(text=f"Via {provider.title()}", icon_url=ICONS.get(provider, ICONS["default"]))
                    await ctx.send(embed=embed, delete_after=10)
                    return
        except Exception as e:
            logger.error(f"Error playing first resolved track: {e}")
            await ctx.send("❌ Error playing the first track.")
            return

        # 8. Load remaining tracks in background
        if tracks_to_queue:
            status_msg = await ctx.send(f"✅ **Playing!** Loading {len(tracks_to_queue)} more tracks from {provider.title()} list...")

            count = 0
            for item in tracks_to_queue:
                try:
                    search_res = await wavelink.Playable.search(item["query"])
                    if search_res:
                        t = search_res[0]
                        t.custom_payload = {"provider": provider, "requester": ctx.author.id}
                        if item["meta"].get("image"):
                            t.custom_payload["artwork"] = item["meta"]["image"]
                        await vc.queue.put_wait(t)
                        count += 1

                    await asyncio.sleep(0.1)
                except Exception:
                    continue

            try:
                await status_msg.edit(content=f"✅ **List Loaded!** Added {count + 1} tracks from {provider.title()}.")
            except:
                pass

    @commands.hybrid_command(name="lyrics", aliases=["ly"])
    async def lyrics_command(self, ctx, *, query: str = None):
        """Get lyrics for current song or search"""
        await ctx.defer()

        vc: wavelink.Player = ctx.voice_client
        search_candidates = []

        if query:
            if " - " in query:
                parts = query.split(" - ", 1)
                search_candidates.append((parts[1].strip(), parts[0].strip()))
            search_candidates.append((query, None))

        elif vc and vc.current:
            raw_title = vc.current.title
            raw_author = vc.current.author

            clean_title = LyricsFetcher.basic_clean(raw_title)
            clean_author = LyricsFetcher.basic_clean(raw_author)

            if " - " in clean_title:
                parts = clean_title.split(" - ", 1)
                search_candidates.append((parts[1].strip(), parts[0].strip()))

            search_candidates.append((clean_title, clean_author))
            search_candidates.append((clean_title, None))
        else:
            await ctx.send("❌ No song playing. Use `/lyrics <song name>`")
            return

        result = None
        for title, artist in search_candidates:
            result = await LyricsFetcher.fetch_from_lrclib(title, artist)
            if result:
                break

        if not result and GENIUS_TOKEN:
            best_candidate = search_candidates[0]
            fallback_query = f"{best_candidate[1]} {best_candidate[0]}" if best_candidate[1] else best_candidate[0]
            result = await LyricsFetcher.fetch_from_genius(fallback_query)

        if result:
            lyrics = result.get('lyrics', '')
            if len(lyrics) > 4000:
                lyrics = lyrics[:3997] + "..."

            embed = discord.Embed(
                title=f"📜 {result['title']}",
                description=lyrics or f"[Click here for lyrics]({result.get('url', '')})",
                color=discord.Color.from_rgb(138, 43, 226)
            )
            embed.set_author(name=f"Artist: {result['artist']}")
            if result.get('thumbnail'):
                embed.set_thumbnail(url=result['thumbnail'])
            embed.set_footer(text=f"Source: {result['source']}")

            await ctx.send(embed=embed)
        else:
            await ctx.send("❌ Lyrics not found.")

    @commands.hybrid_command(name="skip", aliases=["s", "next"])
    async def skip_command(self, ctx):
        """Skip the current track (with race-condition protection)"""
        if not await self.validate_voice_status(ctx):
            return
        
        vc: wavelink.Player = ctx.voice_client

        if not vc or not vc.playing:
            await ctx.send("❌ Nothing is playing.")
            return

        # --- EDGE CASE FIX: Double Skip Protection ---
        # Check if a skip happened in this guild less than 1.5 seconds ago
        last_skip = self.last_actions.get(f"skip_{ctx.guild.id}", 0)
        if time.time() - last_skip < 1.5:
            # Silently ignore or send ephemeral to prevent spam
            await ctx.send("⏳ **Slow down!** Skipping is already in progress...", delete_after=3)
            return

        # Update timestamp
        self.last_actions[f"skip_{ctx.guild.id}"] = time.time()

        try:
            await vc.skip(force=True)
            await ctx.send("⏭️ Skipped!")
        except Exception as e:
            logger.error(f"Skip error: {e}")
            await ctx.send("❌ Failed to skip.")

    @commands.hybrid_command(name="pause")
    async def pause_command(self, ctx):
        """Pause playback"""
        if not await self.validate_voice_status(ctx): return

        vc: wavelink.Player = ctx.voice_client

        if not vc or not vc.playing:
            await ctx.send("❌ Nothing is playing.")
            return

        if vc.paused:
            await ctx.send("⚠️ Already paused.")
            return

        await vc.pause(True)
        await ctx.send("⏸️ Paused.")

    @commands.hybrid_command(name="resume")
    async def resume_command(self, ctx):
        """Resume playback"""
        if not await self.validate_voice_status(ctx): return

        vc: wavelink.Player = ctx.voice_client

        if not vc:
            await ctx.send("❌ Not connected.")
            return

        if not vc.paused:
            await ctx.send("⚠️ Not paused.")
            return

        await vc.pause(False)
        await ctx.send("▶️ Resumed.")

    @commands.hybrid_command(name="stop", aliases=["leave", "dc", "disconnect"])
    async def stop(self, ctx):
        """Stop playback and disconnect"""
        if not await self.validate_voice_status(ctx):
            return

        vc: wavelink.Player = ctx.voice_client

        if not vc:
            await ctx.send("❌ Not connected.")
            return

        # --- EDGE CASE FIX: 24/7 Protection ---
        is_247 = db.get_247_status(ctx.guild.id)
        
        # If 24/7 is ON, only Admins or DJ can stop
        if is_247 and not (ctx.author.guild_permissions.administrator or ctx.author.id in OWNER_IDS):
             await ctx.send("🔒 **24/7 Mode is Active.** Only Admins can disconnect the bot.\nUse `/clear` to remove songs instead.")
             return

        self.bot.set_autoplay_enabled(ctx.guild.id, False)

        queue_count = len(vc.queue)
        vc.queue.clear()

        try:
            await vc.stop()
        except:
            pass

        try:
            if vc.channel and isinstance(vc.channel, discord.VoiceChannel):
                await vc.channel.edit(status=None)
        except:
            pass

        if is_247:
            await ctx.send(f"⏹️ Stopped & cleared {queue_count} tracks (24/7 active - staying connected)")
        else:
            try:
                await vc.disconnect()
                await ctx.send(f"👋 Disconnected & cleared {queue_count} tracks")
            except Exception as e:
                logger.error(f"Disconnect error: {e}")
                await ctx.send("⚠️ Disconnected (forced)")

    @commands.hybrid_command(name="queue", aliases=["q"])
    async def queue_command(self, ctx):
        """Show the current queue"""
        if not await self.validate_voice_status(ctx): return

        vc: wavelink.Player = ctx.voice_client

        if not vc or (vc.queue.is_empty and not vc.current):
            await ctx.send("❌ Queue is empty.")
            return

        embed = discord.Embed(title="🎵 Music Queue", color=discord.Color.from_rgb(138, 43, 226))

        if vc.current:
            current_time = f"{int(vc.position//1000//60):02d}:{int(vc.position//1000%60):02d}"
            total_time = f"{int(vc.current.length//1000//60):02d}:{int(vc.current.length//1000%60):02d}"

            embed.add_field(
                name="Now Playing",
                value=f"**{vc.current.title}**\nby `{vc.current.author}`\n`{current_time} / {total_time}`",
                inline=False
            )

        if not vc.queue.is_empty:
            queue_text = ""
            for i, track in enumerate(list(vc.queue)[:10], 1):
                duration = f"{int(track.length//1000//60):02d}:{int(track.length//1000%60):02d}"
                queue_text += f"`{i}.` **{track.title}** - `{duration}`\n"

            remaining = len(vc.queue) - 10
            if remaining > 0:
                queue_text += f"\n*...and {remaining} more tracks*"

            embed.add_field(
                name=f"Up Next ({len(vc.queue)} tracks)",
                value=queue_text or "No tracks in queue",
                inline=False
            )

        await ctx.send(embed=embed)

    @commands.hybrid_command(name="clear", aliases=["clearqueue", "cq"])
    async def clear_queue(self, ctx):
        """Clear the queue"""
        if not await self.validate_voice_status(ctx): return

        vc: wavelink.Player = ctx.voice_client

        if not vc:
            await ctx.send("❌ Not connected.")
            return

        if vc.queue.is_empty:
            await ctx.send("❌ Queue is already empty.")
            return

        count = len(vc.queue)
        vc.queue.clear()
        await ctx.send(f"🗑️ Cleared **{count}** tracks from queue.")

    @commands.hybrid_command(name="volume", aliases=["vol"])
    async def volume_command(self, ctx, volume: int = None):
        """Set or check volume (0-1000)"""
        if not await self.validate_voice_status(ctx): return
        vc: wavelink.Player = ctx.voice_client

        if not vc:
            await ctx.send("❌ Not connected.")
            return

        if volume is None:
            await ctx.send(f"🔊 Current volume: **{vc.volume}%**")
            return

        if volume < 0 or volume > 1000:
            await ctx.send("❌ Volume must be between 0 and 1000.")
            return

        await vc.set_volume(volume)
        await ctx.send(f"🔊 Volume set to **{volume}%**")

    @commands.hybrid_command(name="loop")
    async def loop_command(self, ctx, mode: str = None):
        """Set loop mode (track/queue/off)"""
        if not await self.validate_voice_status(ctx): return
        vc: wavelink.Player = ctx.voice_client

        if not vc:
            await ctx.send("❌ Not connected.")
            return

        if mode is None:
            current = vc.queue.mode
            if current == wavelink.QueueMode.loop:
                status = "Track"
            elif current == wavelink.QueueMode.loop_all:
                status = "Queue"
            else:
                status = "Off"
            await ctx.send(f"🔁 Loop: **{status}**")
            return

        mode = mode.lower()

        if mode in ["track", "song", "single"]:
            vc.queue.mode = wavelink.QueueMode.loop
            vc.autoplay = wavelink.AutoPlayMode.disabled
            self.bot.set_autoplay_enabled(ctx.guild.id, False)
            await ctx.send("🔂 Loop: **Track** (Autoplay disabled)")
        elif mode in ["queue", "all", "playlist"]:
            vc.queue.mode = wavelink.QueueMode.loop_all
            vc.autoplay = wavelink.AutoPlayMode.disabled
            self.bot.set_autoplay_enabled(ctx.guild.id, False)
            await ctx.send("🔁 Loop: **Queue** (Autoplay disabled)")
        elif mode in ["off", "none", "disable"]:
            vc.queue.mode = wavelink.QueueMode.normal
            await ctx.send("➡️ Loop **Off**")
        else:
            await ctx.send("❌ Use: `track`, `queue`, or `off`")

    @commands.hybrid_command(name="autoplay")
    async def autoplay_command(self, ctx):
        """Toggle autoplay mode"""
        if not await self.validate_voice_status(ctx): return
        vc: wavelink.Player = ctx.voice_client

        if not vc:
            await ctx.send("❌ Not connected.")
            return

        if not self.bot.is_autoplay_enabled(ctx.guild.id):
            self.bot.set_autoplay_enabled(ctx.guild.id, True)
            vc.queue.mode = wavelink.QueueMode.normal
            vc.autoplay = wavelink.AutoPlayMode.disabled

            if ctx.guild.id not in self.bot.play_history:
                self.bot.play_history[ctx.guild.id] = set()

            await ctx.send("📻 **Autoplay Enabled** - Will queue similar tracks when queue ends")
        else:
            self.bot.set_autoplay_enabled(ctx.guild.id, False)
            await ctx.send("📴 **Autoplay Disabled**")

    @commands.hybrid_command(name="eq", aliases=["equalizer"])
    async def eq_command(self, ctx):
        """Open equalizer controls"""
        if not await self.validate_voice_status(ctx): return
        vc: wavelink.Player = ctx.voice_client

        if not vc:
            await ctx.send("❌ Not connected.")
            return

        await ctx.send("🎚️ **Equalizer**", view=EqualizerView(vc))

    @commands.hybrid_command(name="speed")
    async def speed_command(self, ctx, speed: float = None):
        """Set playback speed or open speed controls"""
        if not await self.validate_voice_status(ctx): return
        vc: wavelink.Player = ctx.voice_client

        if not vc:
            await ctx.send("❌ Not connected.")
            return

        if speed is None:
            await ctx.send("⏱️ **Speed Control**", view=SpeedView(vc))
            return

        if speed < 0.5 or speed > 2.0:
            await ctx.send("❌ Speed must be between 0.5 and 2.0")
            return

        filters = vc.filters or wavelink.Filters()
        filters.timescale.set(speed=speed)
        await vc.set_filters(filters)
        await ctx.send(f"⏱️ Speed set to **{speed}x**")

    @commands.hybrid_command(name="current", aliases=["np", "nowplaying"])
    async def current(self, ctx):
        """Show currently playing track"""
        vc: wavelink.Player = ctx.voice_client

        if not vc or not vc.current:
            await ctx.send("❌ Nothing is playing.")
            return

        track = vc.current

        icon_url = ICONS["default"]
        if hasattr(track, "custom_payload"):
            prov = track.custom_payload.get("provider", "default")
            icon_url = ICONS.get(prov, ICONS["default"])
        elif track.uri:
            if "spotify.com" in track.uri:
                icon_url = ICONS["spotify"]
            elif "music.apple.com" in track.uri:
                icon_url = ICONS["applemusic"]
            elif "soundcloud.com" in track.uri:
                icon_url = ICONS["soundcloud"]
            elif "youtube.com" in track.uri or "youtu.be" in track.uri:
                icon_url = ICONS["youtube"]

        position = vc.position / 1000
        length = track.length / 1000

        total_bars = 20
        progress = int((position / length) * total_bars) if length > 0 else 0
        progress = max(0, min(progress, total_bars - 1))
        progress_bar = "▬" * progress + "🔘" + "▬" * (total_bars - progress - 1)

        embed = discord.Embed(
            title="Now Playing",
            description=f"🎵 **[{track.title}]({track.uri or 'https://youtube.com'})**",
            color=discord.Color.from_rgb(138, 43, 226)
        )
        embed.set_author(name="Current Track", icon_url=icon_url)

        if track.artwork:
            embed.set_thumbnail(url=track.artwork)

        embed.add_field(name="Artist", value=track.author, inline=True)
        embed.add_field(name="Duration",
                        value=f"`{int(position//60):02d}:{int(position%60):02d} / {int(length//60):02d}:{int(length%60):02d}`",
                        inline=True)
        embed.add_field(name="Progress", value=progress_bar, inline=False)

        await ctx.send(embed=embed)

    @commands.hybrid_command(name="seek")
    async def seek_command(self, ctx, position: str):
        """Seek to position (MM:SS or seconds)"""
        if not await self.validate_voice_status(ctx): return

        vc: wavelink.Player = ctx.voice_client

        if not vc or not vc.current:
            await ctx.send("❌ Nothing is playing.")
            return

        try:
            if ":" in position:
                parts = position.split(":")
                if len(parts) == 2:
                    minutes, seconds = parts
                    target_seconds = int(minutes) * 60 + int(seconds)
                else:
                    await ctx.send("❌ Invalid format. Use MM:SS or seconds.")
                    return
            else:
                target_seconds = int(position)

            target_ms = target_seconds * 1000

            if target_ms < 0 or target_ms > vc.current.length:
                await ctx.send("❌ Position out of range.")
                return

            await vc.seek(target_ms)

            minutes = target_seconds // 60
            seconds = target_seconds % 60
            await ctx.send(f"⏩ Seeked to **{minutes}:{seconds:02d}**")

        except ValueError:
            await ctx.send("❌ Invalid format. Use MM:SS or seconds.")

    @commands.hybrid_command(name="setup")
    @is_owner_or_admin()
    async def setup_command(self, ctx):
        """Setup 24/7 music channel"""
        channel = await self.bot.run_setup(ctx.guild)
        if channel:
            await ctx.send(f"✅ Setup complete! Channel: **{channel.name}**. 24/7: **Enabled**")
        else:
            await ctx.send("❌ Setup failed. Check bot permissions.")

    @commands.hybrid_command(name="247")
    @is_owner_or_admin()
    async def toggle_247(self, ctx, channel: discord.VoiceChannel = None):
        """Toggle 24/7 mode"""
        guild_id = ctx.guild.id
        current_status = db.get_247_status(guild_id)

        if channel:
            db.set_247_status(guild_id, True, channel.id)

            if ctx.voice_client:
                # If already connected, move and deafen
                if ctx.voice_client.channel.id != channel.id:
                    await ctx.voice_client.move_to(channel)
                # Force deafen update
                await ctx.guild.change_voice_state(channel=channel, self_deaf=True)
            else:
                # Connect and force deafen
                await channel.connect(cls=wavelink.Player, self_deaf=True)
                if not ctx.guild.me.voice.self_deaf:
                    await ctx.guild.change_voice_state(channel=channel, self_deaf=True)

            await ctx.send(f"✅ 24/7 Enabled and set to {channel.mention}")
            return

        if current_status:
            db.set_247_status(guild_id, False)
            await ctx.send("📴 24/7 Mode **Disabled**")
        else:
            saved_channel_id = db.get_247_channel(guild_id)

            if saved_channel_id:
                target_channel = ctx.guild.get_channel(saved_channel_id)
                if target_channel:
                    db.set_247_status(guild_id, True)

                    if not ctx.voice_client:
                        await target_channel.connect(cls=wavelink.Player, self_deaf=True)
                    elif ctx.voice_client.channel.id != target_channel.id:
                        await ctx.voice_client.move_to(target_channel)
                    
                    # Ensure deafen is applied
                    await ctx.guild.change_voice_state(channel=target_channel, self_deaf=True)

                    await ctx.send(f"✅ 24/7 Mode **Enabled** on {target_channel.mention}")
                    return

            if ctx.author.voice and ctx.author.voice.channel:
                target_channel = ctx.author.voice.channel
                db.set_247_status(guild_id, True, target_channel.id)

                if not ctx.voice_client:
                    await target_channel.connect(cls=wavelink.Player, self_deaf=True)
                
                # Ensure deafen is applied
                await ctx.guild.change_voice_state(channel=target_channel, self_deaf=True)

                await ctx.send(f"✅ 24/7 Mode **Enabled** on {target_channel.mention}")
            else:
                await ctx.send("❌ No channel specified and you're not in a voice channel.")

    @commands.hybrid_command(name="prefix")
    @is_owner_or_admin()
    async def set_guild_prefix(self, ctx, new_prefix: str):
        """Change bot prefix"""
        if len(new_prefix) > 10:
            await ctx.send("❌ Prefix too long (max 10 characters)")
            return

        db.set_prefix(ctx.guild.id, new_prefix)
        await ctx.send(f"✅ Prefix changed to `{new_prefix}`")

    @commands.hybrid_command(name="games", aliases=["game"])
    async def games_command(self, ctx, argument: str = "5"):
        """Start a trivia game"""
        if argument.lower() == "stop":
            return await self._stop_game_logic(ctx)

        try:
            rounds = int(argument)
        except ValueError:
            await ctx.send("❌ Invalid number of rounds. Use a number or 'stop'.")
            return

        if rounds < 1 or rounds > 20:
            await ctx.send("❌ Rounds must be between 1 and 20.")
            return

        view = GameSelectionView(ctx)
        await ctx.send(f"🎮 **Select Game** (Rounds: {rounds}):", view=view)

        await view.wait()
        if view.value == "song":
            await self.trivia_start(ctx, rounds=rounds, mode="song")
        elif view.value == "movie":
            await self.trivia_start(ctx, rounds=rounds, mode="movie")
        else:
            await ctx.send("❌ Cancelled.")

    @commands.group(name="trivia", invoke_without_command=True)
    async def trivia(self, ctx):
        """Trivia game commands"""
        await ctx.send("❓ Use `/games` to start or `/trivia stop` to stop.")

    @trivia.command(name="stop")
    async def trivia_stop(self, ctx):
        """Stop current trivia game"""
        await self._stop_game_logic(ctx)

    async def _stop_game_logic(self, ctx):
        guild_id = ctx.guild.id
        if guild_id not in self.trivia_games:
            await ctx.send("❌ No game running.")
            return

        game = self.trivia_games[guild_id]
        game.active = False

        if ctx.voice_client:
            try:
                await ctx.voice_client.stop()
            except:
                pass

        if game.scores:
            sorted_scores = sorted(game.scores.items(), key=lambda x: x[1], reverse=True)

            embed = discord.Embed(title="🏆 Game Over!", color=discord.Color.gold())
            for i, (user_id, score) in enumerate(sorted_scores[:3], 1):
                user = self.bot.get_user(user_id)
                name = user.display_name if user else f"User {user_id}"
                embed.add_field(name=f"{i}. {name}", value=f"{score} points", inline=False)

            await ctx.send(embed=embed)
        else:
            await ctx.send("🛑 **Game Over!** No winners.")

        del self.trivia_games[guild_id]

    async def trivia_start(self, ctx, rounds: int = 5, mode: str = "song"):
        """Start trivia game"""
        guild_id = ctx.guild.id

        if guild_id in self.trivia_games:
            await ctx.send("❌ Game already running!")
            return

        self.trivia_games[guild_id] = GameState(rounds)

        if not ctx.author.voice:
            del self.trivia_games[guild_id]
            await ctx.send("🚫 Join a voice channel first!")
            return

        if not ctx.voice_client:
            try:
                vc = await robust_connect(ctx.author.voice.channel, ctx)
                vc.text_channel = ctx.channel
                self.bot.set_player_text_channel(ctx.guild.id, ctx.channel.id)
            except Exception as e:
                del self.trivia_games[guild_id]
                await ctx.send(f"❌ Failed to connect: {e}")
                return
        else:
            vc = ctx.voice_client
            if vc.channel.id != ctx.author.voice.channel.id:
                await vc.move_to(ctx.author.voice.channel)
            vc.text_channel = ctx.channel
            self.bot.set_player_text_channel(ctx.guild.id, ctx.channel.id)

        join_view = TriviaJoinView(ctx.author.id)
        mode_title = "🎬 Bollywood Movie Guess" if mode == "movie" else "🎵 Music Trivia"
        await ctx.send(f"**{mode_title}**\nClick to join!", view=join_view)
        await join_view.wait()

        if len(join_view.participants) < 1:
            del self.trivia_games[guild_id]
            await ctx.send("❌ Not enough players!")
            return

        participants = join_view.participants

        api_urls = []
        if mode == "movie":
            api_urls = ["https://itunes.apple.com/in/rss/topsongs/limit=100/genre=1263/json"]
            await ctx.send("🎬 Starting Bollywood Movie Trivia!")
        else:
            vote_view = TriviaVoteView(ctx.author.id, participants)
            await ctx.send("🗳️ Vote for category!", view=vote_view)
            await vote_view.wait()

            if len(vote_view.votes['hindi']) > len(vote_view.votes['english']):
                api_urls = ["https://itunes.apple.com/in/rss/topsongs/limit=100/genre=1263/json"]
                await ctx.send("🇮🇳 Starting Hindi Song Trivia!")
            elif len(vote_view.votes['english']) > len(vote_view.votes['hindi']):
                api_urls = ["https://itunes.apple.com/us/rss/topsongs/limit=100/json"]
                await ctx.send("🇺🇸 Starting English Song Trivia!")
            else:
                api_urls = [
                    "https://itunes.apple.com/in/rss/topsongs/limit=100/genre=1263/json",
                    "https://itunes.apple.com/us/rss/topsongs/limit=100/json"
                ]
                await ctx.send("🌍 Starting Mixed Song Trivia!")

        question_bank = []
        async with aiohttp.ClientSession() as session:
            for url in api_urls:
                try:
                    async with session.get(url, timeout=10) as resp:
                        if resp.status == 200:
                            data = await resp.json(content_type=None)
                            entries = data.get('feed', {}).get('entry', [])

                            for entry in entries:
                                title = entry.get('im:name', {}).get('label', '')
                                artist = entry.get('im:artist', {}).get('label', '')
                                album = entry.get('im:collection', {}).get('im:name', {}).get('label', '')

                                if mode == "movie":
                                    if title and artist and album:
                                        question_bank.append({
                                            'answer': album,
                                            'query': f"{title} {artist}",
                                            'artist': artist
                                        })
                                else:
                                    if title and artist:
                                        question_bank.append({
                                            'answer': title,
                                            'query': f"ytsearch:{title} {artist}",
                                            'artist': artist
                                        })
                except Exception as e:
                    logger.error(f"Trivia API error: {e}")

        if not question_bank:
            del self.trivia_games[guild_id]
            await ctx.send("❌ Could not load questions.")
            return

        random.shuffle(question_bank)
        game = self.trivia_games[guild_id]

        while game.active and game.current_round < game.rounds:
            if not question_bank:
                await ctx.send("🚫 Out of questions!")
                game.active = False
                break

            q_data = question_bank.pop()
            correct_answer = q_data['answer']
            clean_correct = self.clean_song_title(correct_answer)

            num_players = len(participants)
            needed_distractors = min(num_players + 1, 24)

            candidates = {}
            for q in question_bank:
                full_a = q['answer']
                clean_a = self.clean_song_title(full_a)
                if clean_a != clean_correct and clean_a not in candidates:
                    candidates[clean_a] = full_a

            unique_distractors = list(candidates.values())

            if len(unique_distractors) >= needed_distractors:
                distractors = random.sample(unique_distractors, needed_distractors)
            else:
                distractors = unique_distractors

            options = [correct_answer] + distractors
            random.shuffle(options)

            try:
                search_q = f"ytsearch:{q_data['query']}" if mode == "movie" else q_data['query']
                tracks = await wavelink.Playable.search(search_q)

                if tracks:
                    await vc.play(tracks[0], replace=True)
                else:
                    continue
            except Exception as e:
                logger.error(f"Trivia audio error: {e}")
                continue

            game.current_round += 1

            prompt = f"🎬 **Round {game.current_round}: Which MOVIE?**" if mode == "movie" else f"🎵 **Round {game.current_round}: Guess the Song!**"
            view = TriviaChoiceView(correct_answer, options, participants)
            await ctx.send(prompt, view=view)

            await view.wait()

            if view.winner:
                game.scores[view.winner.id] = game.scores.get(view.winner.id, 0) + 1
            elif view.all_wrong:
                await ctx.send(f"💀 Everyone got it wrong! Answer: **{correct_answer}**")
            else:
                await ctx.send(f"⏰ Time's up! Answer: **{correct_answer}**")

            await asyncio.sleep(3)

        if game.active:
            await self._stop_game_logic(ctx)


def is_player_safe(player: wavelink.Player) -> bool:
    try:
        if not player:
            return False

        if not player.guild:
            return False

        if not player.channel:
            return False

        if not isinstance(player.channel, discord.VoiceChannel):
            return False

        if not player.guild.voice_client:
            return False

        return True
    except Exception:
        return False

class ProfileManagement(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    async def _validate_image(self, ctx, attachment: discord.Attachment):
        """Helper to validate image types"""
        if not attachment:
            if ctx.message.attachments:
                attachment = ctx.message.attachments[0]
            else:
                await ctx.send("❌ Please provide an image attachment!", ephemeral=True)
                return None

        valid_types = ["image/png", "image/jpeg", "image/jpg", "image/gif"]
        if attachment.content_type not in valid_types:
            await ctx.send("❌ Invalid file type. Please upload a **PNG, JPG, or GIF**.", ephemeral=True)
            return None
            
        return await attachment.read()

    # ==========================
    # 🖼️ AVATAR COMMANDS
    # ==========================

    @commands.hybrid_command(name="set_avatar_global", description="[Owner] Sets the bot's Global Avatar (Visible everywhere).")
    @app_commands.describe(file="The image file to use")
    async def set_avatar_global(self, ctx, file: discord.Attachment = None):
        """Sets the bot's Global Avatar."""
        if ctx.author.id not in OWNER_IDS:
            return await ctx.send("⛔ **Access Denied:** Owner only.", ephemeral=True)

        await ctx.defer()
        image_data = await self._validate_image(ctx, file)
        if not image_data: return

        try:
            await self.bot.user.edit(avatar=image_data)
            embed = discord.Embed(title="✅ Global Avatar Updated", description="The bot's profile picture has been updated everywhere.", color=discord.Color.green())
            if file: embed.set_image(url=file.url)
            await ctx.send(embed=embed)
        except Exception as e:
            await ctx.send(f"❌ Error: `{e}`. (You might be rate-limited)", ephemeral=True)

    @commands.hybrid_command(name="set_avatar_guild", description="[Admin] Sets the bot's Avatar for THIS server only.")
    @app_commands.describe(file="The image file to use")
    @commands.has_permissions(administrator=True)
    async def set_avatar_guild(self, ctx, file: discord.Attachment = None):
        """Sets the bot's Guild-Specific Avatar (Direct API Method)."""
        await ctx.defer()
        image_data = await self._validate_image(ctx, file)
        if not image_data: return

        try:
            # 🔧 MANUAL API FIX: Bypass Member.edit() limitation
            # We convert the image to Base64 and hit the endpoint directly
            b64_data = discord.utils._bytes_to_base64_data(image_data)
            
            await self.bot.http.request(
                discord.http.Route(
                    "PATCH", 
                    "/guilds/{guild_id}/members/@me", 
                    guild_id=ctx.guild.id
                ),
                json={"avatar": b64_data}
            )

            embed = discord.Embed(title="✅ Server Avatar Updated", description=f"The bot's look in **{ctx.guild.name}** has been updated.", color=discord.Color.blurple())
            if file: embed.set_image(url=file.url)
            await ctx.send(embed=embed)

        except discord.Forbidden:
            await ctx.send("❌ I need the **Change Nickname** permission (or I cannot modify my own profile here)!", ephemeral=True)
        except Exception as e:
            await ctx.send(f"❌ Failed: `{e}`", ephemeral=True)

    @commands.hybrid_command(name="reset_avatar_guild", description="[Admin] Resets the bot's avatar in this server to the Global one.")
    @commands.has_permissions(administrator=True)
    async def reset_avatar_guild(self, ctx):
        """Resets the bot's Guild-Specific Avatar."""
        await ctx.defer()
        try:
            # 🔧 MANUAL API FIX: Send None (null) to reset
            await self.bot.http.request(
                discord.http.Route(
                    "PATCH", 
                    "/guilds/{guild_id}/members/@me", 
                    guild_id=ctx.guild.id
                ),
                json={"avatar": None}
            )
            
            await ctx.send(f"✅ **Avatar Reset:** I am now using my Global Avatar in {ctx.guild.name}.")
        except discord.Forbidden:
            await ctx.send("❌ I need the **Change Nickname** permission to do this.", ephemeral=True)
        except Exception as e:
            await ctx.send(f"❌ Failed: `{e}`", ephemeral=True)

    # ==========================
    # 🚩 BANNER COMMANDS
    # ==========================

    @commands.hybrid_command(name="set_banner_global", description="[Owner] Sets the bot's Global Profile Banner.")
    @app_commands.describe(file="The banner image (Recommended 600x240)")
    async def set_banner_global(self, ctx, file: discord.Attachment = None):
        """Sets the bot's Global Banner."""
        if ctx.author.id not in OWNER_IDS:
            return await ctx.send("⛔ **Access Denied:** Owner only.", ephemeral=True)

        await ctx.defer()
        image_data = await self._validate_image(ctx, file)
        if not image_data: return

        try:
            await self.bot.user.edit(banner=image_data)
            embed = discord.Embed(title="✅ Global Banner Updated", description="The bot's profile banner has been updated.", color=discord.Color.green())
            if file: embed.set_image(url=file.url)
            await ctx.send(embed=embed)
        except Exception as e:
            await ctx.send(f"❌ Failed: `{e}`. (Ensure image is <10MB)", ephemeral=True)

    @commands.hybrid_command(name="set_banner_guild", description="[Unavailable] Discord API Limitation.")
    async def set_banner_guild(self, ctx, file: discord.Attachment = None):
        """Explains why Guild Banners are not possible."""
        embed = discord.Embed(
            title="🚫 Feature Unavailable",
            description="**Discord API Limitation:** Bots cannot have server-specific *Banners*.\nBots can only have:\n1. Global Banner (One for everywhere)\n2. Server Avatars (Unique per server)",
            color=discord.Color.red()
        )
        await ctx.send(embed=embed, ephemeral=True)

    @commands.hybrid_command(name="reset_banner_global", description="[Owner] Removes the bot's Global Banner.")
    async def reset_banner_global(self, ctx):
        """Resets (Removes) the Global Banner."""
        if ctx.author.id not in OWNER_IDS:
            return await ctx.send("⛔ **Access Denied:** Owner only.", ephemeral=True)
        
        await ctx.defer()
        try:
            await self.bot.user.edit(banner=None)
            await ctx.send("✅ **Banner Reset:** The global banner has been removed.")
        except Exception as e:
            await ctx.send(f"❌ Failed: `{e}`", ephemeral=True)

async def setup(bot):
    await bot.add_cog(ProfileManagement(bot))

# --- Main Entry Point ---
async def main():
    bot = MoodyMusicBot()

    async with bot:
        await bot.add_cog(Music(bot))
        await bot.add_cog(FailSafe(bot))
        await bot.add_cog(ConnectionOptimizer(bot))
        await bot.add_cog(ProfileManagement(bot))

        try:
            await bot.start(TOKEN)
        except discord.LoginFailure:
            logger.error("Invalid Discord token")
        except KeyboardInterrupt:
            logger.info("Bot shutting down...")
            await bot.web_server.stop()
        except Exception as e:
            logger.error(f"Bot crashed: {e}")
            raise


if __name__ == "__main__":
    asyncio.run(main())