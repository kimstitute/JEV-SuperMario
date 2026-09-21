"""Native Windows monitor for a realtime Jev Mario run."""
import argparse, json, pathlib, tkinter as tk
from PIL import Image, ImageTk

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('run_dir'); a=ap.parse_args()
    root=tk.Tk(); root.title('Jev × Mario - realtime'); root.configure(bg='#111'); root.geometry('760x560')
    image_label=tk.Label(root,bg='#000'); image_label.pack(padx=12,pady=12)
    text=tk.Label(root,text='연결 대기',justify='left',anchor='w',font=('Segoe UI',14),fg='white',bg='#222',padx=12,pady=10); text.pack(fill='x',padx=12)
    run=pathlib.Path(a.run_dir)
    def poll():
        try:
            d=json.loads((run/'live_state.json').read_text(encoding='utf-8'))
            f=run/'live_frame.png'
            if f.exists():
                im=Image.open(f).convert('RGB'); im.thumbnail((720,420)); image_label.image=ImageTk.PhotoImage(im); image_label.configure(image=image_label.image)
            text.configure(text=(f"Jev 선택: {d.get('jev_action') or '대기'}    실제 행동: {d.get('current_action','NOOP')}\n"
                                f"월드 X: {d.get('world_x', d.get('summary',{}).get('max_world_x','–'))}    "
                                f"지연: {d.get('latency_s','–')} s    점프: {d.get('jump_phase','–')}\n"
                                f"상태: {d.get('status','running')}    decision: {d.get('decision','–')}    frame: {d.get('frame','–')}"))
        except Exception: pass
        root.after(250,poll)
    poll(); root.mainloop()
if __name__=='__main__': main()
