"""生成不含私人资料的真实 PDF 测试文件，作者：xxx。"""
from io import BytesIO
from pathlib import Path

from reportlab.pdfgen.canvas import Canvas

PAGE_SIZE = (612, 792)
TEXT_ORIGIN = (50, 740)


def pdf_bytes(pages: tuple[str, ...] = ("Original evidence.",)) -> bytes:
    """参数为各页自造文字，空字符串生成空白页；返回独立生成器写出的 PDF 字节。"""
    output = BytesIO()
    canvas = Canvas(output, pagesize=PAGE_SIZE, invariant=1)
    for text in pages:
        paragraph = canvas.beginText(*TEXT_ORIGIN)
        paragraph.textLines(text)
        canvas.drawText(paragraph)
        canvas.showPage()
    canvas.save()
    return output.getvalue()


def write_pdf(path: Path, pages: tuple[str, ...] = ("Original evidence.",)) -> Path:
    """参数为目标路径和各页文字；写入真实 PDF，返回目标路径。"""
    path.write_bytes(pdf_bytes(pages))
    return path
