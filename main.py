import markdown

# Đọc nội dung từ README.md
with open('README.md', 'r', encoding='utf-8') as file:
    markdown_content = file.read()

# Chuyển đổi Markdown sang HTML
html_content = markdown.markdown(markdown_content)

# Lưu nội dung HTML vào một tập tin
with open('content.html', 'w', encoding='utf-8') as file:
    file.write(html_content)

print("Trang HTML đã được tạo thành công!")