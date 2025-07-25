import re

def clean_bangla_qa(text):
    # [] ব্র্যাকেটসহ টেক্সট মুছে ফেল
    text = re.sub(r'\[[^\]]*\]', '', text)

    # এক্সট্রা নিউলাইন রিমুভ
    text = re.sub(r'\n+', '\n', text)

    # প্রশ্নগুলো এক লাইনে: উত্তর সহ
    lines = []
    chunks = text.split('\n')
    i = 0
    while i < len(chunks):
        line = chunks[i].strip()
        if re.match(r'^\d+।', line):
            q = line
            i += 1
            while i < len(chunks) and not chunks[i].startswith('উত্তর:'):
                q += ' ' + chunks[i].strip()
                i += 1
            if i < len(chunks) and chunks[i].startswith('উত্তর:'):
                ans = chunks[i].replace('উত্তর:', '').strip()
                i += 1
                if i < len(chunks) and chunks[i].startswith('ব্যাখ্যা:'):
                    explanation = chunks[i].replace('ব্যাখ্যা:', '').strip()
                    i += 1
                    lines.append(f"{q} উত্তর: {ans} | ব্যাখ্যা: {explanation}")
                else:
                    lines.append(f"{q} উত্তর: {ans}")
        else:
            i += 1
    return '\n'.join(lines)

# Example usage
with open("Data/ocr_text.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

cleaned = clean_bangla_qa(raw_text)
with open("cleaned_qa.txt", "w", encoding="utf-8") as f:
    f.write(cleaned)
