"""Pygame-based renderer for the Wilderness Survival System (WSS).

This renderer opens a dedicated window and draws a colored tile map with
clearly spaced cells, a split HUD for readability, and visual cues:
- Player marker
- Items (water, food, gold) and traders
- Outer border walls (thick frame); the east goal column highlighted
- Impassable tiles (given current resources) overlaid with a red cross

Layout improvements (readability)
---------------------------------
- A concise Controls bar is drawn across the top of the window each frame
  so instructions remain readable even when the legend panel is small.
- The right legend panel now focuses on Status (bars) + Symbols, reducing text
  density on that side.
- Press F1 to toggle an in‑window Help overlay with the full control legend.

Usage
-----
renderer = PygameRenderer(width, height, cell_size=28)
renderer.draw(map_obj, player_obj, info_dict)
renderer.close()
"""

from __future__ import annotations

from typing import Any, Tuple


class PygameRenderer:
    """Lightweight Pygame renderer backend.

    Parameters
    ----------
    width, height : int
        Map dimensions in tiles.
    cell_size : int
        Pixel size of one map cell (square). Larger values increase spacing.
    show_grid : bool
        If True, draws a faint grid over tiles.
    show_legend : bool
        If True, shows a legend panel on the right.
    legend_width : int
        Width in pixels reserved for the legend panel.
    fps : int
        Target FPS for clock tick.
    """

    def __init__(
        self,
        width: int,
        height: int,
        *,
        cell_size: int = 28,
        show_grid: bool = True,
        show_legend: bool = True,
        legend_width: int = 240,
        fps: int = 30,
    ) -> None:
        try:
            import pygame  # noqa: F401
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Pygame is required for the 'pygame' render_mode. Install with 'pip install pygame'."
            ) from e

        import pygame

        self.w = int(width)
        self.h = int(height)
        self.cell = int(cell_size)
        self.show_grid = bool(show_grid)
        self.show_legend = bool(show_legend)
        self.legend_w = int(legend_width) if show_legend else 0
        self.fps = int(fps)

        # Colors (RGB)
        self.COLORS = {
            "bg": (18, 18, 20),
            "grid": (60, 60, 65),
            "border": (220, 220, 230),
            "goal": (255, 215, 0),
            "text": (235, 235, 240),
            # terrains
            "plains": (196, 226, 120),
            "forest": (54, 130, 54),
            "swamp": (90, 120, 90),
            "mountain": (130, 130, 130),
            "desert": (226, 201, 120),
            # overlays
            "impassable": (200, 50, 50),
            "player": (240, 70, 70),
            "trader": (200, 120, 230),
            "gold": (245, 200, 0),
            "water": (70, 200, 240),
            "food": (240, 170, 60),
        }

        # Initialize pygame display
        pygame.init()
        total_w = self.w * self.cell + self.legend_w + 2  # initial guess
        total_h = self.h * self.cell + 2

        # Create a resizable window, starting near the logical size
        flags = pygame.RESIZABLE
        try:
            info = pygame.display.Info()
            screen_w = int(getattr(info, "current_w", 1920))
            screen_h = int(getattr(info, "current_h", 1080))
            init_w = min(total_w, screen_w)
            init_h = min(total_h, screen_h)
        except Exception:
            init_w, init_h = total_w, total_h

        self._flags = flags
        self._fullscreen = False
        self._show_help = False
        self._overlay_trader = None

        # Create the actual window
        self._window = pygame.display.set_mode((init_w, init_h), self._flags)
        pygame.display.set_caption("Wilderness Survival System — Pygame Renderer")

        # Remember base windowed size (for when we exit fullscreen)
        self._base_window_size = (init_w, init_h)

        # Track last window size so we can detect changes
        self._last_window_size = self._window.get_size()

        # Create logical canvas + fonts sized for this window
        self._rebuild_canvas(init_w, init_h)

        self.clock = pygame.time.Clock()
        self._closed = False


    # Terrain name -> color mapping (fallback safe)
    def _terrain_color(self, name: str) -> Tuple[int, int, int]:
        return self.COLORS.get(name, (180, 180, 180))

    def _draw_statbar(
        self,
        pygame,
        label: str,
        cur: int,
        mx: int,
        x: int,
        y: int,
        w: int,
        h: int,
        fill_color: Tuple[int, int, int],
        *,
        font=None,
    ) -> int:
        """Draw a labeled horizontal bar and return the next y position.

        :param str label: Short label (e.g., "STR").
        :param int cur: Current value.
        :param int mx: Maximum value.
        :param int x: Left pixel of the bar area.
        :param int y: Top pixel of the bar area.
        :param int w: Width of the bar area (pixels).
        :param int h: Height of the bar (pixels).
        :param tuple fill_color: RGB color for the filled portion.
        :return: y position after drawing this element (for stacking vertically).
        :rtype: int
        """
        # Label text
        font = font or self.small
        label_surf = font.render(label, True, self.COLORS["text"])
        self.screen.blit(label_surf, (x, y))

        # Bar background and frame
        bar_x = x + 48
        bar_y = y + 2
        bar_w = max(10, w - 56)
        bar_h = max(6, h - 4)
        pygame.draw.rect(self.screen, (40, 40, 44), (bar_x, bar_y, bar_w, bar_h))
        pygame.draw.rect(self.screen, self.COLORS["border"], (bar_x, bar_y, bar_w, bar_h), width=1)

        # Filled portion
        mx = max(1, int(mx))
        frac = max(0.0, min(1.0, float(cur) / float(mx)))
        fill_w = int(round(frac * (bar_w - 2)))
        if fill_w > 0:
            pygame.draw.rect(self.screen, fill_color, (bar_x + 1, bar_y + 1, fill_w, bar_h - 2))

        # Numeric text
        val_text = f"{int(cur)}/{int(mx)}"
        val_surf = font.render(val_text, True, self.COLORS["text"])
        self.screen.blit(val_surf, (bar_x + bar_w - val_surf.get_width(), y))
        return y + h + 4

    def _draw_legend(self,
                     pygame,
                     info: dict[str, Any],
                     x0: int,
                     y0: int,
                     w: int,
                     h: int
                     ) -> None:
        """Draw the side panel (legend + status) using adaptive scaling.

        Ensures the entire content fits the panel by scaling font sizes and bar
        heights, and by splitting the symbols into multiple columns if needed.
        """
        panel = pygame.Rect(x0, y0, w, h)
        pygame.draw.rect(self.screen, (30, 30, 34), panel)
        pygame.draw.rect(self.screen, self.COLORS["border"], panel, width=1)

        pad = 8
        inner_w = max(40, w - 2 * pad)

        # Gather info pieces
        res = info.get("resources", {}) if isinstance(info, dict) else {}
        s_cur = int(res.get("strength", 0)); s_max = int(res.get("max_strength", 100))
        w_cur = int(res.get("water", 0));    w_max = int(res.get("max_water", 100))
        f_cur = int(res.get("food", 0));     f_max = int(res.get("max_food", 100))
        g_cur = int(res.get("gold", 0))
        pos = info.get("position", (None, None))
        terrain = info.get("terrain", None)
        has_trader = bool(info.get("tile_has_trader", False))
        items_here = info.get("tile_items", []) or []
        tile_costs = info.get("tile_costs") or None
        last_usage = info.get("last_usage") or None

        # Symbol legend lines (rendered after status)
        legend_lines = [
            "Symbols:",
            "  @ Player",
            "  R Trader",
            "  $ Gold",
            "  w Water",
            "  % Food",
            "  LGreen-Plains",
            "  Green-Forest",
            "  DGreen-Swamp",
            "  Grey-Mountain",
            "  Yellow-Desert",
            "",
            "Red X = impassable now",
            "Gold wall = goal edge",
        ]

        # Width-clipped blit helper
        def blit_clipped(font, text, color, x, y, max_w):
            s = text
            surf = font.render(s, True, color)
            if surf.get_width() <= max_w:
                self.screen.blit(surf, (x, y))
                return surf.get_height()
            ell = "…"
            while surf.get_width() > max_w and len(s) > 1:
                s = s[:-2] + ell
                surf = font.render(s, True, color)
            self.screen.blit(surf, (x, y))
            return surf.get_height()

        max_size = max(18, min(int(h * 0.18), 64, int(self.cell * 2.0)))
        min_size = 10
        chosen = None  # (font, line_h, bar_h, cols, used_before_symbols)
        col_gutter = 8
        for size in range(max_size, min_size - 1, -1):
            fnt = pygame.font.SysFont(None, size)
            line_h = max(12, int(size + 2))
            bar_h = max(6, int(size * 0.75))

            # Estimate space used by all non-symbol lines
            y = y0 + 10
            # Header
            y += line_h + 4
            # Status label + 3 bars
            y += line_h
            y += (bar_h + 8) * 3
            # GLD + optional lines
            y += line_h
            if pos[0] is not None:
                y += line_h
            if terrain is not None:
                y += line_h
            if has_trader or items_here:
                y += line_h
            if isinstance(tile_costs, dict):
                y += line_h
            if isinstance(last_usage, dict):
                y += line_h
            y += 6  # spacer before symbols

            # Remaining height for symbols
            avail_h = (y0 + h) - y - 6
            if avail_h < line_h:
                # Not enough height even for a single line; try smaller font
                continue

            min_col_w = max(80, int(size * 2.2))
            max_cols_by_width = max(1, inner_w // max(1, (min_col_w)))
            placed = False
            chosen_cols = 1
            if max_cols_by_width < 1:
                max_cols_by_width = 1
            for cols in range(1, max_cols_by_width + 1):
                # Height needed when distributing lines across 'cols'
                import math as _math
                lines_per_col = int(_math.ceil(len(legend_lines) / cols))
                need_h = lines_per_col * line_h
                if need_h <= avail_h:
                    chosen_cols = cols
                    placed = True
                    break
            if placed:
                chosen = (fnt, line_h, bar_h, chosen_cols, y)
                break

        # Fallback
        if chosen is None:
            fnt = pygame.font.SysFont(None, min_size)
            line_h = max(12, int(min_size + 2))
            bar_h = max(6, int(min_size * 0.75))
            # With the smallest font, compute a reasonable columns count using
            # the same minimum column width heuristic.
            min_col_w = max(80, int(min_size * 2.2))
            max_cols_by_width = max(1, inner_w // max(1, (min_col_w)))
            chosen_cols = max(1, max_cols_by_width)
            used_before_symbols = y0 + 10
        else:
            fnt, line_h, bar_h, chosen_cols, used_before_symbols = chosen

        cur_y = y0 + 10
        # Header
        header = f"Legend  diff={info.get('difficulty','?')} step={info.get('step','?')}"
        blit_clipped(fnt, header, self.COLORS["text"], x0 + pad, cur_y, inner_w)
        cur_y += line_h + 4

        # Status label
        blit_clipped(fnt, "Status:", self.COLORS["text"], x0 + pad, cur_y, inner_w)
        cur_y += line_h
        # Bars
        cur_y = self._draw_statbar(pygame, "STR", s_cur, s_max, x0 + pad, cur_y, inner_w, bar_h, self.COLORS["player"], font=fnt)
        cur_y = self._draw_statbar(pygame, "WAT", w_cur, w_max, x0 + pad, cur_y, inner_w, bar_h, self.COLORS["water"], font=fnt)
        cur_y = self._draw_statbar(pygame, "FOD", f_cur, f_max, x0 + pad, cur_y, inner_w, bar_h, self.COLORS["food"], font=fnt)

        # Diagnostics
        cur_y += blit_clipped(fnt, f"GLD {g_cur}", self.COLORS["text"], x0 + pad, cur_y, inner_w)
        if pos[0] is not None:
            cur_y += blit_clipped(fnt, f"Pos ({pos[0]},{pos[1]})", self.COLORS["text"], x0 + pad, cur_y, inner_w)
        if terrain is not None:
            cur_y += blit_clipped(fnt, f"Terrain: {terrain}", self.COLORS["text"], x0 + pad, cur_y, inner_w)
        if has_trader or items_here:
            here = []
            if has_trader:
                here.append("Trader")
            if items_here:
                here += [t.capitalize() for t in items_here]
            cur_y += blit_clipped(fnt, f"Here: {', '.join(here)}", self.COLORS["text"], x0 + pad, cur_y, inner_w)
        if isinstance(tile_costs, dict):
            m_c = int(tile_costs.get("move", 0)); w_c = int(tile_costs.get("water", 0)); f_c = int(tile_costs.get("food", 0))
            cur_y += blit_clipped(fnt, f"Tile cost M/W/F: {m_c}/{w_c}/{f_c}", self.COLORS["text"], x0 + pad, cur_y, inner_w)
        if isinstance(last_usage, dict):
            m_u = int(max(0, last_usage.get("move", 0)))
            w_u = int(max(0, last_usage.get("water", 0)))
            f_u = int(max(0, last_usage.get("food", 0)))
            cur_y += blit_clipped(fnt, f"Last use M/W/F: {m_u}/{w_u}/{f_u}", self.COLORS["text"], x0 + pad, cur_y, inner_w)

        cur_y += 6

        # Render symbols using the SAME font size as the status, distributing
        # across columns if needed. This keeps scaling consistent.
        import math as _math
        avail_h = max(0, (y0 + h) - cur_y - 6)
        cols = max(1, chosen_cols if 'chosen_cols' in locals() else 1)
        lines_per_col = int(_math.ceil(len(legend_lines) / cols))
        col_w = max(54, inner_w // cols)
        for ci in range(cols):
            for ri in range(lines_per_col):
                idx = ci * lines_per_col + ri
                if idx >= len(legend_lines):
                    break
                s = legend_lines[idx]
                # Clip within column width for safety
                x_col = x0 + pad + ci * col_w
                y_line = cur_y + ri * line_h
                if y_line + line_h > y0 + h - 4:
                    break
                # Use blit_clipped to ensure text fits column width
                blit_clipped(fnt, s, self.COLORS["text"], x_col, y_line, max_w=col_w - 2)

    def _draw_help_overlay(self, pygame) -> None:
        """Draw a modal help overlay with the full control legend (F1)."""
        if not self._show_help:
            return
        full_w, full_h = self.screen.get_width(), self.screen.get_height()
        # Dim background
        dim = pygame.Surface((full_w, full_h), pygame.SRCALPHA)
        dim.fill((0, 0, 0, 140))
        self.screen.blit(dim, (0, 0))

        pad = 14
        box_w = min(760, int(full_w * 0.9))
        box_h = min(360, int(full_h * 0.7))
        box_x = (full_w - box_w) // 2
        box_y = (full_h - box_h) // 2
        panel = pygame.Rect(box_x, box_y, box_w, box_h)
        pygame.draw.rect(self.screen, (32, 32, 36), panel)
        pygame.draw.rect(self.screen, self.COLORS["border"], panel, width=2)

        title = self.small.render("Controls & Help (F1 to close)", True, self.COLORS["text"])
        self.screen.blit(title, (box_x + pad, box_y + pad))
        y = box_y + pad + 24

        lines = [
            "Movement: Arrows or WASD",
            "Diagonals: Y U B N   or   Q E Z C",
            "Wait / Rest: '.' (period) or Space",
            "Quit: ESC or Shift+Q (uppercase Q)",
            "Misc: F11 toggle fullscreen; window is resizable",
            "HUD: Right panel shows Status and Symbols",
        ]
        for s in lines:
            surf = self.small.render(s, True, self.COLORS["text"])
            self.screen.blit(surf, (box_x + pad, y))
            y += 22

    def _draw_border_and_grid(self, pygame):
        # Outer border
        map_rect = pygame.Rect(1, 1, self.w * self.cell, self.h * self.cell)
        pygame.draw.rect(self.screen, self.COLORS["border"], map_rect, width=2)
        # East (goal) wall highlight
        ex = 1 + self.w * self.cell - 2
        pygame.draw.line(
            self.screen, self.COLORS["goal"], (ex, 1), (ex, 1 + self.h * self.cell - 2), width=3
        )
        # Optional grid
        if self.show_grid:
            for x in range(self.w + 1):
                X = 1 + x * self.cell
                pygame.draw.line(self.screen, self.COLORS["grid"], (X, 1), (X, 1 + self.h * self.cell))
            for y in range(self.h + 1):
                Y = 1 + y * self.cell
                pygame.draw.line(self.screen, self.COLORS["grid"], (1, Y), (1 + self.w * self.cell, Y))

    def _draw_tile(self, pygame, x: int, y: int, terrain_name: str) -> None:
        cx = 1 + x * self.cell
        cy = 1 + y * self.cell
        rect = pygame.Rect(cx + 1, cy + 1, self.cell - 2, self.cell - 2)
        pygame.draw.rect(self.screen, self._terrain_color(terrain_name), rect)

    def _overlay_impassable(self, pygame, x: int, y: int) -> None:
        # Draw a red X across the tile
        cx = 1 + x * self.cell
        cy = 1 + y * self.cell
        col = self.COLORS["impassable"]
        pygame.draw.line(self.screen, col, (cx + 3, cy + 3), (cx + self.cell - 3, cy + self.cell - 3), 2)
        pygame.draw.line(self.screen, col, (cx + self.cell - 3, cy + 3), (cx + 3, cy + self.cell - 3), 2)

    def _draw_glyph(self, pygame, ch: str, color: Tuple[int, int, int], x: int, y: int) -> None:
        # Draw a centered glyph in a tile
        cx = 1 + x * self.cell
        cy = 1 + y * self.cell
        surf = self.font.render(ch, True, color)
        rect = surf.get_rect(center=(cx + self.cell // 2, cy + self.cell // 2))
        self.screen.blit(surf, rect)

    def _has(self,
             tile,
             kind: str) -> bool:
        for itm in getattr(tile, "items", []):
            if getattr(itm, "itemType", None) == kind:
                return True
        return False

    def draw(self,
             map_obj,
             player_obj,
             info: dict[str, Any] | None = None
             ) -> None:
        """Draw a full frame.

        :param map_obj: The map instance providing ``getTile(x,y)`` and size.
        :param player_obj: The player with ``position`` and current resources.
        :param dict info: Optional info dict with diagnostics (difficulty, step).
        """
        if self._closed:
            return
        import pygame

        # --- Event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

            # We don't rely on VIDEORESIZE anymore; instead we detect size changes
            # every frame. That avoids platform differences.
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_F11:
                    # Toggle fullscreen; window size will change, and we'll
                    # detect that below and rebuild the canvas/fonts.
                    self._fullscreen = not self._fullscreen
                    if self._fullscreen:
                        # Go fullscreen at desktop resolution
                        self._window = pygame.display.set_mode(
                            (0, 0), self._flags | pygame.FULLSCREEN
                        )
                    else:
                        # Return to last windowed size
                        self._window = pygame.display.set_mode(
                            self._base_window_size, self._flags
                        )

                elif event.key == pygame.K_F1:
                    self._show_help = not self._show_help

        # --- Detect window size changes and rebuild canvas/fonts if needed ---
        window_w, window_h = self._window.get_size()
        if (window_w, window_h) != self._last_window_size:
            self._last_window_size = (window_w, window_h)
            self._rebuild_canvas(window_w, window_h)

        # --- Draw on the logical canvas (self.screen) ---

        # Background
        self.screen.fill(self.COLORS["bg"])

        # Determine visibility radius
        try:
            from src import Difficulty
            radius = Difficulty.VISION_RADII.get(str(getattr(map_obj, "difficulty", "medium")), 3)
        except ImportError:
            diff = str(getattr(map_obj, "difficulty", "medium")).lower()
            radius = 3
            if "easy" in diff:
                radius = 4
            elif "hard" in diff:
                radius = 2
            elif "extreme" in diff:
                radius = 1

        px, py = player_obj.position

        # Terrain tiles
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                mx, my = px + dx, py + dy
                sx, sy = dx + radius, dy + radius
                
                if 0 <= mx < map_obj.width and 0 <= my < map_obj.height:
                    name = map_obj.getTile(mx, my).terrain.name
                    self._draw_tile(pygame, sx, sy, name)

        # Overlays: items and trader glyphs
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                mx, my = px + dx, py + dy
                sx, sy = dx + radius, dy + radius
                
                if 0 <= mx < map_obj.width and 0 <= my < map_obj.height:
                    tile = map_obj.getTile(mx, my)
                    # overlay priority: trader > gold > water > food
                    if self._has(tile, "trader"):
                        self._draw_glyph(pygame, "R", self.COLORS["trader"], sx, sy)
                    elif self._has(tile, "gold"):
                        self._draw_glyph(pygame, "$", self.COLORS["gold"], sx, sy)
                    elif self._has(tile, "water"):
                        self._draw_glyph(pygame, "w", self.COLORS["water"], sx, sy)
                    elif self._has(tile, "food"):
                        self._draw_glyph(pygame, "%", self.COLORS["food"], sx, sy)

        # Impassable tiles relative to current resources
        try:
            curS = getattr(player_obj, "currentStrength", player_obj.maxStrength)
            curW = player_obj.currentWater
            curF = player_obj.currentFood
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    mx, my = px + dx, py + dy
                    sx, sy = dx + radius, dy + radius
                    
                    if 0 <= mx < map_obj.width and 0 <= my < map_obj.height:
                        tile = map_obj.getTile(mx, my)
                        if not tile.is_passable(curS, curW, curF):
                            self._overlay_impassable(pygame, sx, sy)
        except Exception:
            pass

        # Player marker (always visible at center of view)
        self._draw_glyph(pygame, "@", self.COLORS["player"], radius, radius)

        # Grid and borders after content so lines are visible
        self._draw_border_and_grid(pygame)

        # Legend panel (draw below the top controls ribbon to avoid overlap)
        if self.show_legend:
            x0 = 1 + self.w * self.cell + 4
            bar_h = 26
            y0 = 1 + bar_h
            height = max(10, self.h * self.cell - bar_h)
            self._draw_legend(pygame, info or {}, x0, y0, self.legend_w - 6, height)

        # Optional Help overlay (F1)
        self._draw_help_overlay(pygame)

        # --- Blit the logical canvas to the window WITHOUT scaling ---

        try:
            window_w, window_h = self._window.get_size()
            screen_w, screen_h = self.screen.get_width(), self.screen.get_height()

            # Fill window background (letterboxing if needed)
            self._window.fill(self.COLORS["bg"])

            # Center the canvas in the window
            offset_x = max(0, (window_w - screen_w) // 2)
            offset_y = max(0, (window_h - screen_h) // 2)

            self._window.blit(self.screen, (offset_x, offset_y))
        except Exception:
            # Fallback: if something goes weird, just blit at (0,0)
            try:
                self._window.blit(self.screen, (0, 0))
            except Exception:
                pass

        # Draw trader overlay if active
        if self._overlay_trader:
            self._render_trading_menu_internal(self._overlay_trader)
            self._overlay_trader = None

        pygame.display.flip()
        self.clock.tick(self.fps)

    def draw_overlay_message(self, text: str) -> None:
        """Draw a centered overlay message on the window."""
        import pygame

        if not self._window: return

        # Render text
        # Use a large-ish font
        font = pygame.font.SysFont(None, 48)
        surf = font.render(text, True, (255, 255, 255))

        # Create a semi-transparent background
        bg = pygame.Surface((surf.get_width() + 20, surf.get_height() + 20))
        bg.fill((0, 0, 0))
        bg.set_alpha(200)

        # Blit text onto bg
        bg.blit(surf, (10, 10))

        # Center on window
        w, h = self._window.get_size()
        x = (w - bg.get_width()) // 2
        y = (h - bg.get_height()) // 2

        self._window.blit(bg, (x, y))
        pygame.display.flip()

    def draw_trading_menu(self, trader) -> None:
        """Schedule the trading menu to be drawn on the next frame."""
        self._overlay_trader = trader

    def _render_trading_menu_internal(self, trader) -> None:
        """Draw the trading menu overlay on the current window (no flip)."""
        import pygame
        if not self._window: return

        font = pygame.font.SysFont(None, 32)
        title_font = pygame.font.SysFont(None, 48)
        
        lines = ["--- TRADER ---"]
        
        # Check if trader has inventory, otherwise fallback (e.g. if old code)
        if hasattr(trader, "getInventory"):
            inventory = trader.getInventory()
            for i, proposal in enumerate(inventory):
                # proposal = [wants, offers]
                # wants = [g, w, f]
                wants = proposal[0]
                offers = proposal[1]
                
                # Create string like "1. Give 5G -> Get 10W 5F"
                pay_parts = []
                if wants[0]>0: pay_parts.append(f"{wants[0]} Gold")
                if wants[1]>0: pay_parts.append(f"{wants[1]} Water")
                if wants[2]>0: pay_parts.append(f"{wants[2]} Food")
                pay_str = ", ".join(pay_parts) if pay_parts else "Nothing"
                
                get_parts = []
                if offers[0]>0: get_parts.append(f"{offers[0]} Gold")
                if offers[1]>0: get_parts.append(f"{offers[1]} Water")
                if offers[2]>0: get_parts.append(f"{offers[2]} Food")
                get_str = ", ".join(get_parts) if get_parts else "Nothing"
                
                lines.append(f"{i+1}. Give {pay_str} -> Get {get_str}")
        else:
             lines.append("(No inventory available)")

        lines.append("")
        lines.append("Press 1-4 to Buy")

        # Render surfaces
        surfaces = []
        w = 0
        h = 0
        
        # Title
        s = title_font.render(lines[0], True, (255, 255, 100))
        surfaces.append(s)
        w = max(w, s.get_width())
        h += s.get_height() + 10
        
        # Items
        for line in lines[1:]:
            s = font.render(line, True, (255, 255, 255))
            surfaces.append(s)
            w = max(w, s.get_width())
            h += s.get_height() + 5
            
        # Background
        bg = pygame.Surface((w + 40, h + 40))
        bg.fill((0, 0, 0))
        bg.set_alpha(180)
        
        # Blit lines
        y_off = 20
        for s in surfaces:
            x_off = (w + 40 - s.get_width()) // 2
            bg.blit(s, (x_off, y_off))
            y_off += s.get_height() + 5
            
        # Center on screen
        win_w, win_h = self._window.get_size()
        final_x = (win_w - bg.get_width()) // 2
        final_y = (win_h - bg.get_height()) // 2
        
        self._window.blit(bg, (final_x, final_y))

    def close(self) -> None:
        if self._closed:
            return
        try:
            import pygame

            pygame.display.quit()
            pygame.quit()
        except Exception:
            pass
        self._closed = True

    def _rebuild_canvas(self,
                        target_w: int | None = None,
                        target_h: int | None = None
                        ) -> None:
        """Recalculate cell size and recreate the logical canvas & fonts
        so that text is rendered at native resolution (no upscaling blur)."""

        import pygame

        # If not given, use current window size
        if target_w is None or target_h is None:
            target_w, target_h = self._window.get_size()

        # Leave some room for borders and legend
        # Compute the largest cell size that fits in the current window
        max_cell_x = (target_w - self.legend_w - 2) // self.w
        max_cell_y = (target_h - 2) // self.h
        new_cell = max(8, min(max_cell_x, max_cell_y))

        self.cell = int(new_cell)

        total_w = self.w * self.cell + self.legend_w + 2
        total_h = self.h * self.cell + 2

        # Recreate logical canvas at this resolution
        self.screen = pygame.Surface((total_w, total_h)).convert_alpha()

        # Recreate fonts at appropriate sizes for the new cell size
        self.font = pygame.font.SysFont(None, max(14, int(self.cell * 0.6)))
        self.small = pygame.font.SysFont(None, max(16, int(self.cell * 0.5)))
