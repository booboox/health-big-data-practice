"""核对原始文件完整性、新增导航、作业预览和个人实验数据位置。"""
from pathlib import Path
from urllib.parse import unquote
from zipfile import ZipFile
from xml.etree import ElementTree as ET
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / '资料核对/原始文件SHA256.json').read_text(encoding='utf-8'))
    for name, digest in manifest['files'].items():
        path = ROOT / name
        assert path.is_file(), f'缺少原始文件：{name}'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, f'原始内容已改变：{name}'

    navigation = json.loads((ROOT / '资料核对/导航文件.json').read_text(encoding='utf-8'))
    links = 0
    for name in navigation:
        path = ROOT / name
        content = path.read_text(encoding='utf-8')
        # Only generated links are checked in previews; quoted original answers are source material.
        if path.name == '作业预览.md':
            content = content.split('\n---\n', 1)[0] + '\n' + '\n'.join(
                re.findall(r'!?\[[^\]]*\]\([^\n]*预览配图[^\n]*\)', content)
            )
        for target in re.findall(r'\]\(([^)]+)\)', content):
            if target.startswith(('http:', 'https:', '#', 'mailto:')):
                continue
            local = (path.parent / unquote(target.split('#', 1)[0])).resolve()
            assert local.is_relative_to(ROOT), f'链接超出仓库：{name}: {target}'
            assert local.exists(), f'失效链接：{name}: {target}'
            links += 1

    # The extracted body must preserve all original text in its original order.
    ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    previews = 0
    for course in ROOT.glob('session*'):
        if not course.is_dir():
            continue
        preview = course / '个人作业/作业预览.md'
        content = preview.read_text(encoding='utf-8')
        source = next(p for p in (course / '个人作业').glob('*.docx') if '已完成 (1)' not in p.name)
        with ZipFile(source) as package:
            body = ET.fromstring(package.read('word/document.xml')).find('w:body', ns)
            original = ''.join(e.text or '' for e in body.iter() if e.tag.endswith('}t'))
        original = ''.join(c for c in original if c.isalnum())
        extracted = iter(c for c in content.split('\n---\n', 1)[1] if c.isalnum())
        for char in original:
            assert any(char == candidate for candidate in extracted), f'正文提取可能遗漏内容：{source.name}'
        previews += 1

    notebook = ROOT / 'session7-线性模型/个人作业/实验/第8讲-实验.ipynb'
    json.loads(notebook.read_text(encoding='utf-8'))
    assert (ROOT / 'session7-线性模型/8. Linear-models/data/heart.csv').is_file()
    print(f'PASS: {len(manifest["files"])} original files unchanged; {links} local links; {previews} preview texts; notebook JSON and data path.')


if __name__ == '__main__':
    main()
