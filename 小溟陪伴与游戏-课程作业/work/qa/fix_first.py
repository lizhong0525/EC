from pathlib import Path
base=Path(__file__).resolve().parents[2]/'stages'
def change(file, old,new):
    s=file.read_text(encoding='utf8')
    if new not in s:
        assert old in s
        file.write_text(s.replace(old,new),encoding='utf8')
src=base/'01-html'
for name,label,old in [('chat','聊天','#conversation'),('games','游戏','#start')]:
    change(src/'index.html',f'<a href="about.html#{name}">',f'<a id="{name}-guide" href="about.html#{name}">')
    change(src/'about.html',f'<a href="index.html{old}">回到首页的'+('对话示例' if name=='chat' else '使用入口'),f'<a href="index.html#{name}-guide">回到首页的{label}指南入口')
change(base/'02-css'/'lobby.css','[id] { scroll-margin-top: 14px; }','''[id] { scroll-margin-top: 14px; }
.rules article:target {
  background: #e3eedf;
  border-inline-start: 4px solid #275540;
  padding-inline: 14px;
}
.rules article:focus-visible { outline: 2px solid #275540; }''')
