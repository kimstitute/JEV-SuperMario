# RGB template provenance

These 16×16 patches were cropped from RGB observations produced by the installed
`gym-super-mario-bros==9.1.0` environment (SMB1 1-1). No level map or enemy RAM was
used to create the visual templates.

`probe_env.py` is a **scripted calibration run**, not Jev policy evaluation.

- ground.png: `runs/probe/0000.png`, crop `(0, 208, 16, 224)`.
- brick.png: `runs/probe/0120.png`, crop `(208, 144, 224, 160)`.
- question.png: `runs/probe/0120.png`, crop `(144, 144, 160, 160)`.

Coordinates are PIL crop boundaries `(left, top, right, bottom)`.
Sprite art belongs to its respective rights holder. These local experimental
fixtures are not an original asset pack or a license to redistribute game assets.

## Expanded appearance dictionary

`scripts/build_visual_dictionary.py` reads the installed ROM's **static CHR
graphics only** and builds foreground-alpha templates. It does not read level
layouts, runtime map buffers, enemy RAM, or future states. Runtime perception
loads PNGs only and compares them against the currently observed RGB screen.

- `goomba_walk.png`: CHR rows `[70,71] [72,73]` (hex), with a horizontally
  mirrored second walking frame in `goomba_walk_mirror.png`.
- `used.png`: `[157,158] [159,15a]`.
- `stair.png`: `[1ab,1ad] [1ac,1ae]`.
- `pipe_top.png`: `[160,161,162,163] [164,165,166,167]`.
- `pipe_body.png`: `[168,169,126,16a]` repeated twice.

Tile assembly reference: the [SMB disassembly's metatile graphics table](https://github.com/pgattic/smb1-disasm/blob/master/main.asm).
The table describes visual tile composition, not the level's layout. Color
values were checked against local RGB captures. Three block palette phases use
highlight colors `(252,160,68)`, `(228,92,16)`, `(136,20,0)` observed in calibration
screens. Runtime matching ignores alpha-zero and sky-colored template pixels.

Goomba's two frames and pipe top/body were checked on actual diagnostic play.
Used blocks and stair shapes are covered by synthetic appearance tests; these
are not claims of an end-to-end Jev run reaching the staircase.
