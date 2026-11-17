import os
import PyPDF2
from datetime import datetime

def pdf_to_txt(pdf_folder, output_folder):
    """
    批量将PDF文件转换为TXT文本文件

    参数:
    pdf_folder: 存放PDF文件的文件夹路径
    output_folder: 输出TXT文件的文件夹路径
    """
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 创建日志文件
    log_file = os.path.join(output_folder, "conversion_log.log")  # 用 .log 后缀，避免被当成论文去抽取
    with open(log_file, 'w', encoding='utf-8') as log:
        log.write(f"PDF转TXT处理日志 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log.write(f"源文件夹: {pdf_folder}\n")
        log.write(f"输出文件夹: {output_folder}\n\n")

        # 统计变量
        total_files = 0
        success_count = 0
        fail_count = 0

        # 遍历PDF文件夹
        for filename in os.listdir(pdf_folder):
            if not filename.lower().endswith('.pdf'):
                continue

            total_files += 1
            pdf_path = os.path.join(pdf_folder, filename)
            txt_filename = f"{os.path.splitext(filename)[0]}.txt"
            txt_path = os.path.join(output_folder, txt_filename)

            try:
                # 读取PDF文件
                with open(pdf_path, 'rb') as pdf_file:
                    reader = PyPDF2.PdfReader(pdf_file)

                    # 提取所有页面文本
                    full_text = ""
                    for page_num, page in enumerate(reader.pages, start=1):
                        text = page.extract_text()
                        if text:
                            full_text += f"=== 第 {page_num} 页 ===\n{text}\n\n"

                    # 保存TXT文件
                    if full_text.strip():
                        with open(txt_path, 'w', encoding='utf-8') as txt_file:
                            txt_file.write(full_text)
                        log.write(f"[成功] {filename} → {txt_filename} (共{len(reader.pages)}页, {len(full_text)}字符)\n")
                        success_count += 1
                    else:
                        log.write(f"[警告] {filename} → 未提取到有效文本\n")
                        fail_count += 1

            except Exception as e:
                log.write(f"[失败] {filename} → 错误: {str(e)}\n")
                fail_count += 1

        # 写入汇总信息
        summary = f"\n处理完成!\n总文件数: {total_files}\n成功: {success_count}\n失败: {fail_count}"
        log.write(summary)
        print(summary)

if __name__ == "__main__":
    # 配置路径 (根据实际情况修改)
    PDF_FOLDER = "data/pdf"      # 存放PDF的文件夹
    OUTPUT_FOLDER = "data/split" # TXT输出文件夹（extract_relations.py 从这里读）

    # 执行转换
    print(f"开始处理 {PDF_FOLDER} 中的PDF文件...")
    pdf_to_txt(PDF_FOLDER, OUTPUT_FOLDER)
    print(f"处理完成! 结果保存在: {OUTPUT_FOLDER}")