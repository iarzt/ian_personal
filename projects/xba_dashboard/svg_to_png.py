from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF, renderPM
from PIL import Image
from os import listdir
from os.path import isfile, join


onlyfiles = [f for f in listdir('./SVG_files') if isfile(join('./SVG_files', f))]

for svg in onlyfiles:

    drawing = svg2rlg('./SVG_files/' + svg)
    image_name = svg.split('.')[0]

    renderPM.drawToFile(drawing, './ballparks/' + image_name + '.png', fmt='PNG')

    img = Image.open('./ballparks/' + image_name + '.png')

    img = img.convert("RGBA")
    datas = img.getdata()


    newData = []
    for item in datas:
        if item[0] == 255 and item[1] == 255 and item[2] == 255:
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)


    img.putdata(newData)

    img.save('./ballparks/' + image_name + '.png', "PNG")
