import pymupdf

# Open PDF
pdf_path = "./sgk-toan-lop-4-tap-1-ket-noi-tri-thuc.pdf"
doc = pymupdf.open(pdf_path)

print(f"Total pages: {len(doc)}")
print(f"Metadata: {doc.metadata}")
print("\n" + "=" * 70)
print("First 3 pages content:")
print("=" * 70)

# Check first 3 pages
for i in range(min(3, len(doc))):
    page = doc[i]
    text = page.get_text()
    print(f"\n--- Page {i+1} ---")
    print(f"Text length: {len(text)} characters")
    print(f"First 500 chars:\n{text[:500]}")
    print(f"\nLast 200 chars:\n{text[-200:]}")

    # Check if page has images
    images = page.get_images()
    print(f"\nNumber of images on page: {len(images)}")

doc.close()
