for f in *imgspage*; do
    mv "$f" "${f//imgspage/imgs_page}"
done
