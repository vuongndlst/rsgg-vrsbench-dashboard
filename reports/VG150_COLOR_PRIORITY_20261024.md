# Màu được ghi trong GT theo đủ 150 lớp VG150 — TRAIN/VAL

Nguồn: `VG150_OBJECT_COLOR_TRAIN_VAL_AUDIT_V2_20261024.json`. Xếp hạng theo TRAIN; VAL trình bày riêng. Không phân tích thuộc tính TEST.

**Tỷ lệ = số object có ít nhất một nhãn màu / toàn bộ object của lớp.** Đây là tần suất được gán nhãn, không phải xác suất object có màu. Thiếu attribute là chưa được ghi nhãn. Bộ H5 chỉ giữ 200 attribute, không đại diện mọi nhãn trong VG gốc.

V2 bổ sung maroon/cream/khaki; giữ audit lexical ban đầu để truy vết. `multicolored` và `colorful` đếm riêng, không tự quy đổi thành màu cụ thể. Tổng histogram là số lần xuất hiện nhãn, không phải số object.

| Split | Objects | Có nhãn màu | Tỷ lệ | >=2 màu | 11 màu cơ bản / tổng lần xuất hiện nhãn |
|---|---:|---:|---:|---:|---:|
| train | 670591 | 180556 | 26.92% | 35895 | 93.91% |
| val | 62754 | 14245 | 22.70% | 2059 | 92.66% |

11 màu cơ bản: black, white, gray, red, green, blue, yellow, brown, orange, pink, purple. Màu mở rộng: tan, beige, maroon, cream, khaki. Silver/gold cần phân biệt màu và bề ngoài kim loại; blond dành riêng cho tóc.

Dấu * chỉ nhóm pilot đề xuất, chưa phải whitelist đã kiểm định. Chỉ số entropy nhãn và mẫu số theo attribute có trong CSV; không dùng tỷ lệ này để đoán màu từ tên lớp.

| Hạng TRAIN | Lớp | TRAIN có màu / tổng | TRAIN % | VAL có màu / tổng | VAL % | Ba màu TRAIN thường gặp | Pilot |
|---:|---|---:|---:|---:|---:|---|:---:|
| 1 | shirt | 14421/21672 | 66.54 | 847/1336 | 63.40 | white:4423; blue:3153; black:2601 | * |
| 2 | cat | 1393/2190 | 63.61 | 35/55 | 63.64 | black:696; white:634; brown:363 | * |
| 3 | jacket | 4129/6505 | 63.47 | 313/477 | 65.62 | black:1495; blue:815; red:576 | * |
| 4 | pant | 5070/8095 | 62.63 | 308/496 | 62.10 | black:1899; white:881; blue:690 | * |
| 5 | toilet | 795/1288 | 61.72 | 26/45 | 57.78 | white:742; brown:16; blue:11 | * |
| 6 | short | 3050/5018 | 60.78 | 94/179 | 52.51 | black:983; white:749; blue:623 | * |
| 7 | dog | 1668/2745 | 60.77 | 44/90 | 48.89 | brown:770; white:760; black:695 | * |
| 8 | coat | 1460/2479 | 58.89 | 147/218 | 67.43 | black:517; blue:248; red:213 | * |
| 9 | bus | 1773/3064 | 57.87 | 120/215 | 55.81 | white:768; red:627; blue:379 | * |
| 10 | bear | 1105/1923 | 57.46 | 8/24 | 33.33 | brown:597; black:287; white:223 |  |
| 11 | train | 2085/3765 | 55.38 | 42/112 | 37.50 | red:652; yellow:617; white:520 | * |
| 12 | laptop | 1015/1854 | 54.75 | 52/110 | 47.27 | black:443; silver:361; white:233 |  |
| 13 | truck | 1449/2650 | 54.68 | 161/269 | 59.85 | white:614; red:322; blue:196 | * |
| 14 | tie | 883/1623 | 54.41 | 25/53 | 47.17 | black:298; red:216; blue:197 |  |
| 15 | helmet | 2094/3855 | 54.32 | 33/92 | 35.87 | black:1016; white:409; blue:345 | * |
| 16 | jean | 1881/3494 | 53.84 | 121/232 | 52.16 | blue:1709; black:136; gray:42 |  |
| 17 | cap | 1308/2453 | 53.32 | 41/82 | 50.00 | black:390; white:382; blue:219 | * |
| 18 | horse | 1662/3130 | 53.10 | 33/78 | 42.31 | brown:1108; white:513; black:333 | * |
| 19 | surfboard | 1184/2270 | 52.16 | 25/55 | 45.45 | white:753; blue:217; yellow:187 |  |
| 20 | sock | 1021/1967 | 51.91 | 23/46 | 50.00 | white:509; black:281; blue:127 |  |
| 21 | hat | 2762/5331 | 51.81 | 114/236 | 48.31 | black:918; white:673; blue:400 | * |
| 22 | cow | 1327/2562 | 51.80 | 24/49 | 48.98 | white:661; brown:625; black:558 | * |
| 23 | plane | 1369/2644 | 51.78 | 42/136 | 30.88 | white:927; blue:377; red:308 |  |
| 24 | plate | 3313/6437 | 51.47 | 210/481 | 43.66 | white:2639; blue:240; brown:203 | * |
| 25 | towel | 854/1725 | 49.51 | 65/111 | 58.56 | white:531; blue:124; red:66 |  |
| 26 | curtain | 896/1815 | 49.37 | 114/236 | 48.31 | white:409; red:116; blue:106 |  |
| 27 | pillow | 1488/3126 | 47.60 | 162/465 | 34.84 | white:760; red:195; blue:184 |  |
| 28 | bowl | 1301/2767 | 47.02 | 93/225 | 41.33 | white:727; blue:151; black:101 |  |
| 29 | umbrella | 1793/3828 | 46.84 | 139/284 | 48.94 | white:427; blue:407; black:399 |  |
| 30 | airplane | 434/934 | 46.47 | 19/37 | 51.35 | white:315; blue:148; red:83 |  |
| 31 | bag | 2072/4491 | 46.14 | 182/373 | 48.79 | black:757; white:425; blue:293 | * |
| 32 | shoe | 3519/7743 | 45.45 | 106/250 | 42.40 | black:1697; white:1553; brown:307 | * |
| 33 | sign | 6172/13679 | 45.12 | 565/1511 | 37.39 | white:3188; red:1647; black:1337 |  |
| 34 | sneaker | 590/1368 | 43.13 | 16/35 | 45.71 | white:373; black:245; blue:63 |  |
| 35 | glove | 1067/2503 | 42.63 | 19/48 | 39.58 | black:654; white:192; brown:126 |  |
| 36 | hair | 4731/11126 | 42.52 | 252/522 | 48.28 | brown:2000; black:1226; blond:1046 |  |
| 37 | letter | 1453/3430 | 42.36 | 102/241 | 42.32 | white:756; black:310; red:173 |  |
| 38 | vase | 635/1538 | 41.29 | 55/184 | 29.89 | white:213; blue:133; black:89 |  |
| 39 | car | 3771/9187 | 41.05 | 1131/2637 | 42.89 | white:970; black:834; red:657 | * |
| 40 | racket | 645/1591 | 40.54 | 10/36 | 27.78 | black:367; white:284; red:210 |  |
| 41 | boot | 725/1794 | 40.41 | 26/57 | 45.61 | black:387; brown:146; white:73 |  |
| 42 | fork | 602/1490 | 40.40 | 41/131 | 31.30 | silver:503; white:72; gray:22 |  |
| 43 | cup | 956/2391 | 39.98 | 79/205 | 38.54 | white:520; red:137; blue:123 |  |
| 44 | flower | 1762/4453 | 39.57 | 204/608 | 33.55 | white:415; yellow:413; pink:350 |  |
| 45 | zebra | 983/2499 | 39.34 | 14/59 | 23.73 | black:890; white:889; brown:56 |  |
| 46 | sink | 667/1707 | 39.07 | 56/133 | 42.11 | white:546; silver:87; black:15 |  |
| 47 | clock | 1117/2862 | 39.03 | 45/130 | 34.62 | white:640; black:480; gold:157 |  |
| 48 | bird | 865/2226 | 38.86 | 19/64 | 29.69 | white:370; black:324; gray:175 |  |
| 49 | chair | 2591/6720 | 38.56 | 473/1321 | 35.81 | brown:593; black:566; white:538 |  |
| 50 | sheep | 805/2116 | 38.04 | 8/29 | 27.59 | white:614; black:160; brown:126 |  |
| 51 | board | 941/2479 | 37.96 | 53/125 | 42.40 | white:410; black:209; blue:144 |  |
| 52 | phone | 586/1545 | 37.93 | 41/104 | 39.42 | black:309; silver:149; white:87 |  |
| 53 | kite | 725/1972 | 36.76 | 9/17 | 52.94 | red:243; blue:240; white:179 |  |
| 54 | pot | 518/1448 | 35.77 | 61/233 | 26.18 | silver:140; white:88; black:85 |  |
| 55 | animal | 500/1420 | 35.21 | 19/61 | 31.15 | brown:209; white:191; black:148 |  |
| 56 | motorcycle | 906/2575 | 35.18 | 50/117 | 42.73 | black:441; red:222; white:158 |  |
| 57 | seat | 910/2603 | 34.96 | 68/198 | 34.34 | black:368; white:174; blue:141 |  |
| 58 | leaf | 2538/7286 | 34.83 | 538/1283 | 41.93 | green:2193; brown:168; yellow:111 |  |
| 59 | elephant | 998/2920 | 34.18 | 30/97 | 30.93 | gray:742; brown:263; black:71 |  |
| 60 | box | 1013/2978 | 34.02 | 97/277 | 35.02 | white:333; brown:206; red:128 |  |
| 61 | vehicle | 598/1778 | 33.63 | 174/420 | 41.43 | white:212; red:109; black:98 |  |
| 62 | skateboard | 776/2374 | 32.69 | 15/48 | 31.25 | black:506; white:96; red:84 |  |
| 63 | paper | 806/2512 | 32.09 | 86/318 | 27.04 | white:635; red:48; brown:46 |  |
| 64 | table | 3375/10640 | 31.72 | 327/1318 | 24.81 | brown:1382; white:1131; black:412 |  |
| 65 | bed | 599/1893 | 31.64 | 51/206 | 24.76 | white:322; blue:94; brown:87 |  |
| 66 | bike | 925/2935 | 31.52 | 62/267 | 23.22 | black:349; red:230; blue:189 |  |
| 67 | handle | 1380/4390 | 31.44 | 96/303 | 31.68 | black:423; silver:310; white:241 |  |
| 68 | tile | 1024/3318 | 30.86 | 65/320 | 20.31 | white:439; brown:177; blue:85 |  |
| 69 | boat | 880/2870 | 30.66 | 76/248 | 30.65 | white:514; blue:224; red:163 |  |
| 70 | flag | 566/1871 | 30.25 | 58/224 | 25.89 | red:281; white:200; blue:199 |  |
| 71 | bench | 976/3258 | 29.96 | 62/288 | 21.53 | brown:335; black:192; green:165 |  |
| 72 | cabinet | 664/2243 | 29.60 | 86/367 | 23.43 | brown:305; white:264; black:41 |  |
| 73 | roof | 1159/3938 | 29.43 | 147/506 | 29.05 | red:236; brown:226; gray:202 |  |
| 74 | basket | 439/1505 | 29.17 | 50/144 | 34.72 | brown:145; black:60; white:59 |  |
| 75 | railing | 510/1755 | 29.06 | 66/220 | 30.00 | black:163; white:95; silver:73 |  |
| 76 | giraffe | 837/2903 | 28.83 | 12/62 | 19.35 | brown:728; white:337; tan:143 |  |
| 77 | tire | 1009/3524 | 28.63 | 136/564 | 24.11 | black:955; white:38; silver:22 |  |
| 78 | vegetable | 305/1074 | 28.40 | 4/31 | 12.90 | green:196; red:36; white:33 |  |
| 79 | banana | 692/2437 | 28.40 | 14/71 | 19.72 | yellow:490; green:190; brown:22 |  |
| 80 | fence | 2081/7399 | 28.13 | 149/525 | 28.38 | black:530; green:375; white:333 |  |
| 81 | logo | 809/2901 | 27.89 | 29/97 | 29.90 | white:308; red:190; blue:155 |  |
| 82 | plant | 893/3348 | 26.67 | 155/625 | 24.80 | green:790; brown:34; red:29 |  |
| 83 | number | 500/1906 | 26.23 | 16/58 | 27.59 | white:175; black:174; red:62 |  |
| 84 | snow | 1466/5598 | 26.19 | 71/333 | 21.32 | white:1421; black:27; green:16 |  |
| 85 | house | 694/2725 | 25.47 | 179/464 | 38.58 | white:342; brown:119; red:80 |  |
| 86 | wave | 822/3298 | 24.92 | 17/85 | 20.00 | white:700; blue:206; green:101 |  |
| 87 | lamp | 564/2310 | 24.42 | 115/445 | 25.84 | white:257; black:104; silver:53 |  |
| 88 | door | 1830/7567 | 24.18 | 302/1121 | 26.94 | white:738; brown:352; red:170 |  |
| 89 | wheel | 1336/5620 | 23.77 | 75/303 | 24.75 | black:896; white:170; red:78 |  |
| 90 | nose | 889/3762 | 23.63 | 16/113 | 14.16 | black:533; brown:148; pink:143 |  |
| 91 | eye | 767/3285 | 23.35 | 27/134 | 20.15 | black:256; brown:206; green:122 |  |
| 92 | screen | 316/1356 | 23.30 | 33/125 | 26.40 | black:154; white:87; blue:58 |  |
| 93 | tail | 1298/5578 | 23.27 | 38/148 | 25.68 | black:491; white:311; blue:218 |  |
| 94 | desk | 316/1375 | 22.98 | 65/312 | 20.83 | brown:183; white:63; black:47 |  |
| 95 | engine | 383/1670 | 22.93 | 6/43 | 13.95 | black:104; red:93; blue:86 |  |
| 96 | pizza | 459/2012 | 22.81 | 0/27 | 0.00 | white:206; brown:114; red:92 |  |
| 97 | food | 746/3281 | 22.74 | 29/184 | 15.76 | white:371; brown:153; green:111 |  |
| 98 | post | 815/3597 | 22.66 | 75/357 | 21.01 | black:214; white:160; gray:121 |  |
| 99 | tower | 360/1614 | 22.30 | 38/159 | 23.90 | white:104; gray:66; brown:64 |  |
| 100 | building | 3756/17118 | 21.94 | 762/4441 | 17.16 | white:1370; brown:736; red:612 |  |
| 101 | pole | 2739/12666 | 21.62 | 230/1212 | 18.98 | black:738; gray:557; white:417 |  |
| 102 | paw | 335/1561 | 21.46 | 4/19 | 21.05 | white:212; black:70; brown:36 |  |
| 103 | fruit | 251/1192 | 21.06 | 9/60 | 15.00 | green:77; orange:66; red:52 |  |
| 104 | counter | 485/2326 | 20.85 | 37/271 | 13.65 | white:182; brown:111; black:87 |  |
| 105 | beach | 441/2127 | 20.73 | 13/78 | 16.67 | brown:234; tan:91; white:69 |  |
| 106 | bottle | 722/3580 | 20.17 | 32/281 | 11.39 | green:225; brown:123; white:119 |  |
| 107 | ski | 461/2307 | 19.98 | 9/58 | 15.52 | black:183; red:130; white:122 |  |
| 108 | stand | 303/1543 | 19.64 | 40/130 | 30.77 | black:95; brown:77; white:46 |  |
| 109 | tree | 5487/28976 | 18.94 | 621/3568 | 17.40 | green:4525; brown:724; white:176 |  |
| 110 | rock | 780/4533 | 17.21 | 57/314 | 18.15 | gray:461; brown:194; white:92 |  |
| 111 | hill | 404/2417 | 16.71 | 16/124 | 12.90 | green:259; white:90; brown:67 |  |
| 112 | drawer | 218/1352 | 16.12 | 46/259 | 17.76 | white:113; brown:78; black:10 |  |
| 113 | trunk | 516/3223 | 16.01 | 39/203 | 19.21 | brown:290; gray:135; black:39 |  |
| 114 | wire | 397/2489 | 15.95 | 20/131 | 15.27 | black:196; white:86; blue:35 |  |
| 115 | book | 392/2464 | 15.91 | 69/453 | 15.23 | white:105; black:84; red:80 |  |
| 116 | wing | 414/2616 | 15.83 | 13/76 | 17.11 | white:152; gray:81; black:66 |  |
| 117 | light | 1257/8188 | 15.35 | 133/905 | 14.70 | red:352; white:274; yellow:181 |  |
| 118 | orange | 204/1387 | 14.71 | 2/35 | 5.71 | orange:153; green:25; yellow:23 |  |
| 119 | mountain | 384/2743 | 14.00 | 29/180 | 16.11 | white:179; green:98; brown:72 |  |
| 120 | ear | 1007/7611 | 13.23 | 14/184 | 7.61 | black:298; brown:278; white:255 |  |
| 121 | shelf | 351/3012 | 11.65 | 50/435 | 11.49 | white:139; brown:84; black:55 |  |
| 122 | sidewalk | 605/5267 | 11.49 | 114/1281 | 8.90 | gray:357; white:89; red:85 |  |
| 123 | face | 622/5421 | 11.47 | 14/162 | 8.64 | white:335; black:207; brown:85 |  |
| 124 | street | 649/6081 | 10.67 | 136/1365 | 9.96 | gray:348; black:162; white:106 |  |
| 125 | head | 1330/13055 | 10.19 | 38/354 | 10.73 | black:448; white:346; brown:346 |  |
| 126 | glass | 650/6658 | 9.76 | 33/539 | 6.12 | black:312; blue:66; brown:66 |  |
| 127 | branch | 352/3612 | 9.75 | 14/358 | 3.91 | brown:193; green:109; gray:22 |  |
| 128 | room | 125/1346 | 9.29 | 19/209 | 9.09 | white:59; brown:23; black:20 |  |
| 129 | track | 370/4284 | 8.64 | 15/144 | 10.42 | brown:177; gray:64; black:63 |  |
| 130 | girl | 319/3752 | 8.50 | 24/218 | 11.01 | blond:117; white:60; black:42 |  |
| 131 | player | 197/2406 | 8.19 | 11/41 | 26.83 | white:101; black:38; blue:37 |  |
| 132 | woman | 1232/15372 | 8.01 | 104/1504 | 6.91 | blond:349; white:271; black:249 |  |
| 133 | boy | 307/4375 | 7.02 | 18/207 | 8.70 | blond:79; white:68; black:61 |  |
| 134 | man | 2327/33535 | 6.94 | 153/2385 | 6.42 | black:810; white:741; blue:307 |  |
| 135 | skier | 92/1348 | 6.82 | 0/36 | 0.00 | white:27; black:26; blue:14 |  |
| 136 | leg | 892/13112 | 6.80 | 37/509 | 7.27 | white:304; black:244; brown:149 |  |
| 137 | child | 122/1836 | 6.64 | 7/107 | 6.54 | blond:39; white:25; black:17 |  |
| 138 | lady | 78/1308 | 5.96 | 6/111 | 5.41 | blond:30; white:16; black:14 |  |
| 139 | window | 1282/21705 | 5.91 | 251/7000 | 3.59 | white:662; black:277; brown:100 |  |
| 140 | hand | 612/10798 | 5.67 | 9/303 | 2.97 | black:347; white:144; red:40 |  |
| 141 | mouth | 127/2258 | 5.62 | 5/62 | 8.06 | black:43; pink:26; white:20 |  |
| 142 | neck | 161/2875 | 5.60 | 6/63 | 9.52 | white:64; brown:51; black:28 |  |
| 143 | kid | 55/1020 | 5.39 | 5/37 | 13.51 | blond:17; white:12; black:8 |  |
| 144 | guy | 58/1115 | 5.20 | 3/49 | 6.12 | black:23; white:18; blue:7 |  |
| 145 | person | 1055/20966 | 5.03 | 71/1923 | 3.69 | black:312; white:251; blue:163 |  |
| 146 | men | 45/1136 | 3.96 | 1/55 | 1.82 | white:17; black:12; gray:6 |  |
| 147 | arm | 175/5874 | 2.98 | 23/228 | 10.09 | black:52; white:28; brown:20 |  |
| 148 | windshield | 41/1990 | 2.06 | 2/126 | 1.59 | black:19; white:7; blue:5 |  |
| 149 | people | 141/8147 | 1.73 | 9/708 | 1.27 | white:44; black:30; green:20 |  |
| 150 | finger | 21/1739 | 1.21 | 1/60 | 1.67 | white:12; pink:3; brown:3 |  |

## Diễn giải trước khi chọn lớp

- Quần áo/phụ kiện và xe: tần suất cao và màu có ích để phân biệt instance; phải kiểm tra mask, kích thước và màu logo/nền.
- Động vật: giữ đa màu khi có bằng chứng; không ép một màu duy nhất.
- Toilet/plate có tần suất cao nhưng màu thường ít biến thiên; dùng làm đối chứng thiên lệch prior, không tự gán white.
- Person/man/woman: không gán màu quần áo/da/tóc cho toàn người. Gắn màu vào node shirt/hat/hair tương ứng; không suy ra sắc tộc.
- Đủ 150 lớp được báo cáo; lớp thiếu mẫu validation không được coi là đã kiểm định. Mọi instance lỗi/unknown giữ trong mẫu số phù hợp.
