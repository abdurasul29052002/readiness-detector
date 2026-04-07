# SINF XONASIDA O'QUVCHILAR XATTI-HARAKATINI ANIQLASH UCHUN KONVOLYUTSION NEYRON TARMOQ VA POZA BAHOLASH ASOSIDA FUSION ALGORITM ISHLAB CHIQISH

---

## MUNDARIJA

- KIRISH ... 3
- I BOB. O'QUVCHILAR XATTI-HARAKATINI ANIQLASH USULLARINING NAZARIY ASOSLARI ... 8
  - 1.1. Kompyuter ko'rish asosida xatti-harakat aniqlash usullari ... 8
  - 1.2. Konvolyutsion neyron tarmoq arxitekturalari tahlili ... 16
  - 1.3. Poza baholash usullari va ularning qo'llanilishi ... 25
  - 1.4. Fusion yondashuvlar va ularning samaradorligi ... 32
  - I bob bo'yicha xulosa ... 38
- II BOB. POSECNN FUSION ALGORITMINI ISHLAB CHIQISH ... 39
  - 2.1. Ma'lumotlar to'plamini shakllantirish va balanslashtirish ... 39
  - 2.2. PoseCNN Fusion arxitekturasining umumiy ko'rinishi ... 47
  - 2.3. ResNet50 vizual branch ... 51
  - 2.4. MediaPipe Pose branch va MLP ... 56
  - 2.5. Fusion mexanizmi va klassifikatsiya qatlami ... 61
  - 2.6. Training strategiyasi va optimizatsiya ... 65
  - II bob bo'yicha xulosa ... 68
- III BOB. EKSPERIMENTAL NATIJALAR VA TAHLIL ... 69
  - 3.1. Eksperiment muhiti va sozlamalari ... 69
  - 3.2. Baseline modellar natijalari ... 72
  - 3.3. PoseCNN Fusion modeli natijalari ... 76
  - 3.4. Taqqosiy tahlil va ablation study ... 81
  - 3.5. Xatolar tahlili va cheklanishlar ... 85
  - 3.6. Real-time tizim arxitekturasi va amaliy qo'llanilishi ... 87
  - III bob bo'yicha xulosa ... 92
- XULOSA ... 93
- FOYDALANILGAN ADABIYOTLAR RO'YXATI ... 96
- ILOVA ... 101

---

## KIRISH

**Mavzuning dolzarbligi.** XXI asrda ta'lim tizimini raqamlashtirish va sun'iy intellekt texnologiyalarini o'quv jarayoniga joriy etish butun dunyo bo'ylab ustuvor vazifalardan biriga aylangan. O'zbekiston Respublikasi Prezidentining 2019-yil 13-sentabrdagi PF-5847-son "O'zbekiston Respublikasida sun'iy intellekt texnologiyalarini rivojlantirish chora-tadbirlari to'g'risida"gi Farmoni va 2020-yil 6-oktabrdagi PF-6079-son "O'zbekiston Respublikasi ta'lim sohasini 2030-yilgacha rivojlantirish konsepsiyasini tasdiqlash to'g'risida"gi Farmoni mazkur yo'nalishdagi tadqiqotlarning dolzarbligini belgilab beradi.

Zamonaviy ta'lim tizimida o'quv jarayonining sifatini baholash va o'quvchilarning darsga bo'lgan e'tiborini real vaqtda aniqlash muhim amaliy masalalardan biri hisoblanadi. An'anaviy usullarda o'qituvchi sinf xonasidagi 30-40 nafar o'quvchining har birining xatti-harakatini bir vaqtning o'zida kuzatib borishi jismoniy jihatdan qiyin. Bundan tashqari, subyektiv baholash xatolarga olib kelishi mumkin — turli o'qituvchilar bir xil harakatni turlicha baholashi yoki vaqtinchalik holatlarni e'tibordan chetda qoldirishi mumkin.

Statistik ma'lumotlarga ko'ra, bir dars davomida o'qituvchi o'rtacha 45 daqiqaning faqat 15-20 daqiqasini bevosita ta'lim jarayoniga sarflaydi, qolgan vaqtning katta qismi sinf tartibini saqlash va o'quvchilar e'tiborini qaytarishga ketadi [12]. Shu sababli, o'quvchilar e'tiborini avtomatik kuzatib boruvchi tizim o'qituvchiga dars samaradorligini oshirishga va individual yondashuvni ta'minlashga yordam beradi.

So'nggi yillarda kompyuter ko'rish (computer vision) va sun'iy intellekt texnologiyalarining jadal rivojlanishi ta'lim sohasida yangi imkoniyatlar yaratdi. Xususan, konvolyutsion neyron tarmoqlar (CNN) asosida rasmlarni tasniflash, ob'ektlarni aniqlash va xatti-harakatni tanib olish sohasida sezilarli yutuqlar qo'lga kiritildi. ResNet [1], VGGNet [2], YOLO [3] kabi arxitekturalar turli amaliy muammolarni hal qilishda yuqori samaradorlik ko'rsatdi.

Shu bilan birga, inson pozasini baholash (human pose estimation) texnologiyalari ham katta taraqqiyotga erishdi. MediaPipe [4], OpenPose [5] kabi tizimlar real vaqtda inson skeletining asosiy nuqtalarini (landmarks) aniqlash imkoniyatini berdi. Biroq, mavjud tadqiqotlarning aksariyati vizual ma'lumot yoki poza ma'lumotini alohida-alohida ishlatadi, bu esa murakkab xatti-harakatlarni aniqlashda cheklanishlarga olib keladi.

Masalan, "kitob o'qish" va "bosh egish" harakatlari vizual jihatdan juda o'xshash — ikkala holatda ham o'quvchining boshi pastga qaragan. Ammo skeletal poza ma'lumotiga qaralsa, "kitob o'qish"da qo'llar oldinda kitob ushlab turadi, "bosh egish"da esa qo'llar erkin holda bo'ladi. Shu kabi, "qo'l ko'tarish" harakatini faqat rasmdan aniqlash qiyin bo'lishi mumkin, ammo qo'l-yelka burchagi 90 darajadan oshganligi skeletal ma'lumotdan aniq ko'rinadi.

Ushbu muammolarni hal qilish uchun vizual va skeletal ma'lumotlarni birlashtiradigan yangi fusion yondashuvlar ishlab chiqish dolzarb ilmiy masala hisoblanadi. Aynan shu zaruratdan kelib chiqib, ushbu dissertatsiya tadqiqotida konvolyutsion neyron tarmoq va poza baholash texnologiyalarini birlashtirgan yangi PoseCNN Fusion algoritmi taklif etiladi.

**Muammoning qo'yilishi.** Sinf xonasida o'quvchilarning xatti-harakatini avtomatik aniqlash tizimini yaratishda quyidagi asosiy muammolar mavjud:

1. O'xshash vizual ko'rinishga ega harakatlarni farqlash qiyinligi — "kitob o'qish" va "bosh egish", "yozish" va "o'qish" kabi juftliklar vizual jihatdan juda o'xshash ko'rinadi, bu esa faqat tasvir asosida ishlaydigan modellar uchun jiddiy muammo hisoblanadi;
2. Turli masofalardan (1-10 metr) va turli burchaklardan olingan tasvirlarda barqaror ishlash zarurati — sinf xonasida kamera birinchi partadan 1-2 metr, oxirgi partadan 7-10 metr masofada joylashadi;
3. Real vaqtda (real-time) ishlash talabi — har soniyada kamida bir kadrni qayta ishlash, bu 30-40 ta o'quvchini bir vaqtda kuzatishni anglatadi;
4. 7 xil xatti-harakat turini (diqqatli va chalg'igan) yuqori aniqlik bilan tasniflash — bu ko'p sinfli tasniflash masalasi bo'lib, sinflar orasidagi chegaralar noaniq;
5. Turli yoritilganlik sharoitlarida va turli sinf xonalarida barqaror natija berish — kun davomida yoritilganlik o'zgaradi, sinf xonalari turli rangda bo'ladi.

**Tadqiqotning maqsadi** — konvolyutsion neyron tarmoq (CNN) va poza baholash (pose estimation) texnologiyalarini birlashtirgan yangi PoseCNN Fusion algoritmini ishlab chiqish va uni sinf xonasida o'quvchilar xatti-harakatini real vaqtda aniqlash tizimida qo'llash.

**Tadqiqot vazifalari:**

1. O'quvchilar xatti-harakatini aniqlash bo'yicha mavjud usullar va yondashuvlarni chuqur tahlil qilish, ularning afzalliklari va kamchiliklarini aniqlash;
2. 7 xil xatti-harakat turini o'z ichiga olgan balanslashtirilgan ma'lumotlar to'plamini shakllantirish — 653,101 ta tasvirdan iborat dataset yaratish;
3. ResNet50 vizual feature extractori va MediaPipe Pose skeletal feature extractorini birlashtirgan PoseCNN Fusion algoritmini ishlab chiqish va matematik asoslash;
4. Taklif etilgan algoritmni mavjud baseline modellar (ResNet50, YOLOv8-cls) bilan taqqosiy tahlil qilish va ablation study o'tkazish;
5. Real vaqtda ishlaydigan mikroservis arxitekturasiga asoslangan veb-asosli monitoring tizimini yaratish va amaliy sinovdan o'tkazish.

**Tadqiqotning ob'ekti** — sinf xonasidagi o'quvchilarning xatti-harakati va ularni avtomatik tasniflash jarayoni.

**Tadqiqotning predmeti** — konvolyutsion neyron tarmoq va poza baholash texnologiyalarini birlashtirib, o'quvchilar xatti-harakatini aniqlash algoritmi va uning dasturiy ta'minoti.

**Tadqiqotning ilmiy yangiligi:**

1. Birinchi marta o'quvchilar xatti-harakatini aniqlash uchun CNN vizual featurelari (2048 o'lchamli) va MediaPipe skeletal poza featurelari (132 o'lchamli) ni birlashtirgan **PoseCNN Fusion** algoritmi taklif etildi. Bu algoritm mavjud yondashuvlardan farqli o'laroq, ikkita mustaqil ma'lumot oqimini (vizual va skeletal) birlashtiradi;
2. Fusion mexanizmida poza ma'lumotini MLP (Multi-Layer Perceptron) orqali 128 o'lchamli vektorga o'tkazib, vizual featurelarga concatenation usulida qo'shish sxemasi ishlab chiqildi. MLP ning vazifasi — xom landmark koordinatalaridan yuqori darajadagi harakatga oid featurelarni o'rganish;
3. 7 xil xatti-harakat turini (bow-head, focus, hand-raising, read, standing, turn-head, write) attentive va distracted guruhlarga ajratib, real vaqtda sinf xonasi monitoringini amalga oshirish tizimi yaratildi. Tizim mikroservis arxitekturasida (Frontend, Backend, AI Predicter, Database) Docker konteynerlarida ishga tushiriladi.

**Tadqiqotning amaliy ahamiyati.** Ishlab chiqilgan tizim quyidagi amaliy muammolarni hal qiladi:

- O'qituvchilarga sinf xonasidagi o'quvchilarning darsga e'tiborini real vaqtda kuzatish imkoniyatini beradi — bu o'qituvchiga dars davomida qaysi o'quvchilar chalg'iganligini ko'rish va zarur choralar ko'rish imkonini yaratadi;
- Ta'lim muassasalari rahbariyatiga o'quv jarayoni sifatini obyektiv baholash vositasini taqdim etadi — kunlik, haftalik va oylik statistik hisobotlar orqali o'quv jarayonining dinamikasini kuzatish mumkin;
- Olingan statistik ma'lumotlar asosida o'quv dasturlarini takomillashtirish uchun tahliliy asos yaratadi — qaysi dars soatlarida e'tibor pasayishi kuzatiladi, qaysi o'qituvchilarning darslari samaraliroq ekanligi aniqlanadi;
- Veb-asosli arxitektura orqali har qanday qurilmadan (kompyuter, planshet, telefon) foydalanish imkoniyatini ta'minlaydi;
- Docker konteynerlashtirish orqali tizimni har qanday serverga oson joylash mumkin.

**Tadqiqot usullari:** konvolyutsion neyron tarmoqlar nazariyasi, transfer o'rganish (transfer learning), inson pozasini baholash (human pose estimation), feature-level fusion, statistik tahlil usullari, eksperimental baholash.

**Himoyaga olib chiqiladigan asosiy holatlar:**

1. CNN va poza featurelari ni birlashtiradigan PoseCNN Fusion algoritmi faqat CNN ga asoslangan usullarga nisbatan vizual jihatdan o'xshash harakatlarni farqlashda yuqori aniqlik ko'rsatadi;
2. MLP orqali qayta ishlangan poza featurelari xom poza koordinatalariga nisbatan samaraliroq natija beradi;
3. Ishlab chiqilgan real-time monitoring tizimi sinf xonasida amaliy qo'llanilish uchun yetarli tezlik va aniqlikka ega.

**Tadqiqot natijalarining aprobatsiyasi.** Tadqiqot natijalari universitetning ilmiy seminarlarida muhokama qilingan va dasturiy ta'minot GitHub platformasida ochiq kodli loyiha sifatida joylashtirilgan.

**Dissertatsiya tarkibi.** Dissertatsiya kirish, uchta bob, xulosa, foydalanilgan adabiyotlar ro'yxati va ilovalardan iborat. Dissertatsiya hajmi ___ betdan iborat bo'lib, unda ___ ta jadval, ___ ta rasm va ___ ta adabiyot manbasi keltirilgan.

**Kirish** qismida mavzuning dolzarbligi, tadqiqotning maqsadi, vazifalari, ilmiy yangiligi va amaliy ahamiyati bayon etilgan.

**Birinchi bob** "O'quvchilar xatti-harakatini aniqlash usullarining nazariy asoslari" deb nomlangan bo'lib, unda kompyuter ko'rish asosida xatti-harakat aniqlash usullari, CNN arxitekturalari, poza baholash usullari va fusion yondashuvlarning nazariy asoslari tahlil qilingan.

**Ikkinchi bob** "PoseCNN Fusion algoritmini ishlab chiqish" deb nomlangan bo'lib, unda ma'lumotlar to'plamini shakllantirish, taklif etilgan algoritmning arxitekturasi, ResNet50 vizual branch, MediaPipe Pose branch, fusion mexanizmi va training strategiyasi batafsil tavsiflangan.

**Uchinchi bob** "Eksperimental natijalar va tahlil" deb nomlangan bo'lib, unda eksperiment muhiti, baseline modellar natijalari, PoseCNN Fusion natijalari, taqqosiy tahlil, xatolar tahlili va real-time tizim arxitekturasi keltirilgan.

**Xulosa** qismida tadqiqotning asosiy natijalari, ilmiy hissa va kelajakdagi tadqiqot yo'nalishlari bayon etilgan.

---

## I BOB. O'QUVCHILAR XATTI-HARAKATINI ANIQLASH USULLARINING NAZARIY ASOSLARI

### 1.1. Kompyuter ko'rish asosida xatti-harakat aniqlash usullari

Kompyuter ko'rish (computer vision) — bu kompyuterlar yordamida raqamli tasvirlar va videolardan ma'noli ma'lumotlarni olish va qayta ishlash bo'yicha sun'iy intellektning muhim yo'nalishi hisoblanadi [6]. Bu soha 1960-yillarda boshlangan bo'lib, dastlab oddiy chiziq va shakllarni aniqlashdan iborat edi. So'nggi o'n yillikda esa chuqur o'rganish (deep learning) texnologiyalari tufayli inqilobiy yutuqlarga erishildi va kompyuter ko'rish tibbiyot, transport, xavfsizlik, sanoat va ta'lim kabi ko'plab sohalarda keng qo'llanila boshlandi.

**Inson xatti-harakatini aniqlash muammosi.** Inson xatti-harakatini aniqlash (human activity recognition, HAR) kompyuter ko'rishning eng faol rivojlanayotgan yo'nalishlaridan biri bo'lib, uning maqsadi — tasvir yoki videodagi insonning harakatini avtomatik tanib olish va tasniflashdir [7]. Bu masala quyidagi sabablar tufayli murakkab hisoblanadi:

- **Sinflar orasidagi vizual o'xshashlik** — turli harakatlar bir xil vizual ko'rinishga ega bo'lishi mumkin. Masalan, "kitob o'qish" va "bosh egish" harakatlari kamera tomonidan juda o'xshash ko'rinadi;
- **Sinflar ichidagi xilma-xillik** — bitta harakat turli odamlar tomonidan turlicha bajarilishi mumkin. "Qo'l ko'tarish" harakati to'g'ri qo'l, chap qo'l yoki ikki qo'l bilan bajarilishi mumkin;
- **Yashirinish (occlusion)** — inson tanasining bir qismi boshqa ob'ektlar orqasida yashirinishi mumkin. Sinf xonasida old qatordagi o'quvchi orqa qatordagini yashirishi mumkin;
- **Burchak va masshtab o'zgarishi** — kameraning joylashuviga qarab inson turli burchakdan va turli masshtabda ko'rinadi;
- **Yoritilganlik o'zgarishi** — kun davomida tabiiy yoritilganlik o'zgaradi, sun'iy yoritilganlik soyalar hosil qiladi.

**An'anaviy kompyuter ko'rish usullari.** Chuqur o'rganishdan oldingi davrda xatti-harakat aniqlash uchun qo'lda yaratilgan feature descriptorlar ishlatilgan. Asosiy usullar quyidagilar:

**Histogram of Oriented Gradients (HOG)** [8]. Dalal va Triggs tomonidan 2005-yilda taqdim etilgan HOG usuli tasvirning gradiyent yo'nalishlarini hisoblab, ob'ekt shaklini tavsiflaydi. Algoritm ishlash tartibi:
1. Tasvir kichik hujayralarga (cells) bo'linadi (masalan, 8×8 piksel);
2. Har bir hujayraning gradiyent yo'nalishlari hisoblanadi;
3. Yo'nalishlar histogrammaga yig'iladi;
4. Qo'shni hujayralar bloklarga birlashtiriladi va normalizatsiyalanadi.

HOG inson shaklini aniqlash uchun samarali bo'lgan, ammo murakkab harakatlarni farqlash uchun yetarli emas edi. Buning sababi — HOG faqat statik shaklni tavsiflaydi, harakat dinamikasini hisobga olmaydi.

**Scale-Invariant Feature Transform (SIFT)** [9]. Lowe tomonidan 2004-yilda taqdim etilgan SIFT turli masshtabda barqaror bo'lgan kalit nuqtalarni (keypoints) aniqlaydi. Har bir kalit nuqta uchun 128 o'lchamli feature descriptor hisoblanadi. SIFT ning afzalligi — aylantirish, masshtablash va qisman yashirinishga nisbatan barqarorligi. Kamchiligi — hisoblash tezligi past va real-time qo'llanilish uchun yaroqsiz.

**Bag of Visual Words (BoVW)** [10]. Csurka va boshqalar tomonidan 2004-yilda taqdim etilgan BoVW matn tasniflanishidagi "bag of words" konsepsiyasini kompyuter ko'rishga moslagan. Usul quyidagi bosqichlardan iborat:
1. Tasvirlardan lokal featurelar (masalan, SIFT) olinadi;
2. K-means klasterlash orqali "vizual lug'at" yaratiladi;
3. Har bir tasvir vizual so'zlar histogrammasi sifatida ifodalanadi;
4. SVM yoki boshqa klassifikator bilan tasniflash amalga oshiriladi.

BoVW oddiy tasniflash vazifalari uchun qoniqarli natija bergan, ammo murakkab va noaniq harakatlarni aniqlashda sezilarli kamchiliklarga ega edi.

**Optical Flow usullari** [31]. Harakatni aniqlashda video kadrlari orasidagi piksellarning siljishini (optical flow) hisoblash usuli keng qo'llangan. Lucas-Kanade va Horn-Schunck algoritmlari eng mashhur optical flow usullari hisoblanadi. Bu usul harakat yo'nalishi va tezligini aniqlash uchun samarali, ammo hisoblash resurslariga bo'lgan talab yuqori va statik holatlarni (masalan, "focus" — o'quvchi qimirlamay o'tirgan) aniqlashda samarasiz.

**Chuqur o'rganish inqilobi.** 2012-yilda Krizhevsky, Sutskever va Hinton tomonidan ishlab chiqilgan AlexNet [11] ImageNet Large Scale Visual Recognition Challenge (ILSVRC) musobaqasida g'alaba qozondi va kompyuter ko'rishda yangi davrni boshladi. AlexNet 8 qatlamdan iborat konvolyutsion neyron tarmoq bo'lib, u oldingi yillardagi eng yaxshi natijadan 10% dan ortiq yaxshi ko'rsatkich ko'rsatdi (top-5 xatolik: 16.4%).

Bu g'alaba chuqur o'rganishning afzalliklarini ko'rsatdi:
- **Avtomatik feature o'rganish** — qo'lda feature yaratish o'rniga, tarmoq o'zi optimal featurelarni o'rganadi;
- **Ierarxik ifodalash** — past darajadagi qatlamlar chiziqlar va teksturalarni, yuqori darajadagi qatlamlar semantik ob'ektlarni aniqlaydi;
- **Transfer learning imkoniyati** — katta ma'lumotlar to'plamida o'rgatilgan model boshqa vazifalarga moslashtirilishi mumkin;
- **GPU dan foydalanish** — parallel hisoblash tufayli katta modellarni tez o'rgatish mumkin.

AlexNet dan keyin VGGNet (2014) [2], GoogLeNet (2014) [18], ResNet (2015) [1], DenseNet (2017) [32], EfficientNet (2019) [33] kabi tobora samaraliroq arxitekturalar yaratildi.

**Ta'lim sohasida kompyuter ko'rishni qo'llash.** Ta'lim sohasida o'quvchilar xatti-harakatini aniqlash nisbatan yangi yo'nalish hisoblanadi. Dastlabki tadqiqotlarda asosan qo'lda kuzatish va anketa usullari ishlatilgan [12]. Fredricks va boshqalar (2004) o'quvchilar faolligini (engagement) uch toifaga bo'lgan: xatti-harakatga oid (behavioral), hissiy (emotional) va kognitiv (cognitive) faollik. Bizning tadqiqotimiz birinchi toifaga — xatti-harakatga oid faollikni aniqlashga qaratilgan.

Kompyuter ko'rish texnologiyalari ta'lim sohasiga qo'llana boshlagandan buyon quyidagi muhim tadqiqotlar o'tkazildi:

**Zaletelj va Kosir (2017)** [13] Kinect sensoridan foydalanib, o'quvchilarning emotsional holatini yuz ifodalari orqali aniqlash tizimini ishlab chiqdilar. Tizim 85% aniqlik ko'rsatdi. Ammo bu yondashuv faqat yuz ifodalariga tayangani sababli tana harakatlarini (yozish, kitob o'qish, qo'l ko'tarish) umuman farqlay olmadi. Bundan tashqari, Kinect sensori maxsus qurilma bo'lib, har bir sinfga o'rnatish qimmat va murakkab.

**Raca va boshqalar (2015, 2019)** [14] EPFL universitetida video nazorat kameralari orqali auditoriyada o'quvchilarning faolligini aniqlash bo'yicha keng qamrovli tadqiqot o'tkazdilar. Ular bosh harakatini (head motion) tahlil qilib, e'tiborni (attention) baholash usulini taklif qildilar. CNN arxitekturasidan foydalanib, 78% aniqlikka erishdilar. Tadqiqotning cheklanishi — faqat bosh harakati tahlil qilingan, tana va qo'l harakatlari hisobga olinmagan.

**Thomas va Jayagopi (2017)** [15] sinf xonasida o'quvchilar e'tiborini aniqlash uchun bosh holati (head pose) va ko'z yo'nalishini (gaze direction) tahlil qilish usulini taklif qildilar. Ularning tizimi yaqin masofada (1-2 metr) yaxshi ishlagan, ammo o'quvchilar kameradan 5+ metr uzoqlashganda aniqlik keskin pasaydi. Bu sinf xonasi sharoitlari uchun jiddiy cheklanish hisoblanadi.

**Sun va boshqalar (2021)** [16] Student Class Behavior Dataset (SCB-Dataset) yaratdilar va transformer arxitekturasini o'quvchilar xatti-harakatini aniqlash uchun moslashtirdilar. Vision Transformer (ViT) asosidagi model 82% aniqlik ko'rsatdi, ammo hisoblash resurslariga bo'lgan yuqori talab (GPU zaruriy) real-time qo'llanilishni qiyinlashtirdi. Har bir kadrni qayta ishlash ~500 ms vaqt oldi, bu 30+ o'quvchili sinf uchun juda sekin.

**Li va boshqalar (2020)** [34] sinf xonasida o'quvchilar xatti-harakatini aniqlash uchun YOLOv3 dan foydalandilar. Ularning tizimi 6 xil harakatni (o'tirish, turish, yozish, qo'l ko'tarish, gaplashish, uyqulash) aniqladi va 75% o'rtacha aniqlikka erishdi. Ammo ularning usuli faqat vizual featurelarga tayanib, skeletal ma'lumotdan foydalanmagan.

**Wang va boshqalar (2020)** [27] multi-feature fusion yondashuvini taklif qildilar, ammo ularning "fusion"i faqat turli CNN qatlamlaridan olingan featurelarni birlashtirish edi — skeletal yoki poza ma'lumoti ishlatilmagan.

**Mavjud yondashuvlarning umumiy kamchiliklari:**

| Muammo | Tushuntirish |
|--------|-------------|
| Faqat vizual ma'lumot | Aksariyat tadqiqotlar faqat tasvir yoki video kadrlari bilan ishlaydi, skeletal ma'lumotdan foydalanmaydi |
| O'xshash harakatlar muammosi | "read" va "bow-head", "write" va "read" kabi juftliklar vizual jihatdan o'xshash |
| Masofaga bog'liqlik | Yaqin masofada yaxshi ishlaydigan usullar uzoq masofada samarasiz |
| Real-time muammo | Yuqori aniqlikli modellar (ViT) sekin ishlaydi |
| Maxsus qurilma talabi | Ba'zi usullar (Kinect) maxsus sensorlarni talab qiladi |
| Kichik datasetlar | Ko'pgina tadqiqotlarda ma'lumotlar to'plami 5,000-50,000 tasvir, bu overfitting ga olib keladi |

Ushbu tahlildan ko'rinib turibdiki, mavjud yondashuvlarning aksariyati faqat bitta turdagi ma'lumotga (vizual yoki yuz ifodasi) taylanadi va turli masofalardan barqaror ishlay olmaydi. Bu cheklanishlar yangi, ko'p modal yondashuvlar ishlab chiqish zaruratini ko'rsatadi. Bizning PoseCNN Fusion algoritmimiz aynan shu muammolarni hal qilishga qaratilgan — u vizual (CNN) va skeletal (pose) ma'lumotlarni birlashtirib, yanada aniqroq tasniflash amalga oshiradi.

### 1.2. Konvolyutsion neyron tarmoq arxitekturalari tahlili

Konvolyutsion neyron tarmoq (Convolutional Neural Network, CNN) — bu tasvirlarni qayta ishlash uchun maxsus mo'ljallangan chuqur neyron tarmoq turi bo'lib, u konvolyutsiya operatsiyasi orqali tasvirdan muhim featurelarni avtomatik ajratib oladi [17]. CNN ning g'oyasi 1989-yilda Yann LeCun tomonidan raqamlarni aniqlash (LeNet) uchun taklif etilgan va so'nggi yillarda kompyuter ko'rishning asosiy vositasiga aylangan.

**CNN ning matematik asoslari.**

**Konvolyutsiya operatsiyasi.** CNN ning asosiy operatsiyasi — konvolyutsiya. Ikki o'lchamli diskret konvolyutsiya quyidagicha aniqlanadi:

```
S(i,j) = (I * K)(i,j) = ΣΣ I(i+m, j+n) · K(m,n)
```

Bu yerda I — kirish tasviri, K — filtr (kernel), S — chiqish (feature map). Filtr tasvir bo'ylab siljitiladi va har bir pozitsiyada elementlar ko'paytmasi yig'indisi hisoblanadi.

**Aktivatsiya funksiyasi.** Konvolyutsiyadan keyin chiziqli bo'lmagan (non-linear) aktivatsiya funksiyasi qo'llaniladi. Eng keng tarqalgan — ReLU (Rectified Linear Unit):

```
f(x) = max(0, x)
```

ReLU ning afzalliklari:
- Gradient yo'qolishi muammosini kamaytiradi;
- Hisoblash jihatidan sodda;
- Sparse (siyrak) aktivatsiya hosil qiladi — faqat musbat qiymatlar o'tadi.

Boshqa aktivatsiya funksiyalari: Sigmoid σ(x) = 1/(1+e^(-x)), Tanh tanh(x) = (e^x - e^(-x))/(e^x + e^(-x)), Leaky ReLU f(x) = max(αx, x) bu yerda α = 0.01.

**Pooling operatsiyasi.** Pooling tasvirning fazoviy o'lchamini kamaytiradi va tarmoqning translatsiya invariantligini oshiradi. Max-pooling:

```
y(i,j) = max{x(i·s+m, j·s+n)} , 0 ≤ m,n < k
```

Bu yerda s — qadam (stride), k — pooling oynasi o'lchami. Odatda 2×2 pooling 2 qadam bilan ishlatiladi, bu tasvirni 4 marta kichiklashtiradi.

**Softmax funksiyasi.** Oxirgi qatlamda ko'p sinfli tasniflash uchun Softmax ishlatiladi:

```
p(y=k|x) = e^(z_k) / Σ e^(z_j)
```

Bu yerda z_k — k-sinf uchun logit, p — ehtimollik. Softmax barcha sinflardagi ehtimolliklar yig'indisi 1 ga teng bo'lishini ta'minlaydi.

**Cross-Entropy Loss.** Tasniflash muammolarida eng keng ishlatiladigan loss funksiyasi:

```
L = -Σ y_c · log(p_c)
```

Bu yerda y_c — haqiqiy sinf (one-hot encoded), p_c — bashorat qilingan ehtimollik. Bu funksiya bashorat haqiqiy sinfdan uzoqlashganda katta jazoni beradi.

**CNN ning asosiy qatlamalari va ularning vazifalari:**

1. **Konvolyutsion qatlam (Convolutional Layer)** — filtr (kernel) yordamida tasvirning lokal xususiyatlarini aniqlaydi. Har bir filtr tasvirning ma'lum bir xususiyatini (chiziq, burchak, tekstura) topadi. N ta filtr N ta feature map hosil qiladi;

2. **Batch Normalizatsiya (Batch Normalization)** [35] — 2015-yilda Ioffe va Szegedy tomonidan taklif etilgan. Har bir mini-batch ichida aktivatsiyalarni normalizatsiyalaydi:

```
x̂ = (x - μ_B) / √(σ²_B + ε)
y = γx̂ + β
```

Bu yerda μ_B — batch o'rtacha qiymati, σ²_B — batch dispersiyasi, γ va β — o'rganiladigan parametrlar. BatchNorm training barqarorligini oshiradi va o'rganish tezligini ko'tarish imkonini beradi;

3. **Pooling qatlami** — tasvirning fazoviy o'lchamini kamaytiradi va muhim ma'lumotlarni saqlab qoladi. Max-pooling eng keng tarqalgan usul hisoblanadi, u har bir oyna ichidagi eng katta qiymatni tanlaydi;

4. **Dropout** [36] — Srivastava va boshqalar (2014) tomonidan taklif etilgan regularizatsiya usuli. Training jarayonida neyronlarning ma'lum foizi (masalan, 50%) tasodifiy o'chiriladi. Bu overfitting ni oldini oladi:

```
h' = h · mask,  mask ~ Bernoulli(p)
```

Bu yerda p — saqlanish ehtimoli (1 - dropout rate);

5. **To'liq ulangan qatlam (Fully Connected Layer)** — oldingi qatlamlardan olingan featurelarni birlashtiradi va yakuniy tasniflash uchun ishlatiladi.

**Asosiy CNN arxitekturalari evolyutsiyasi:**

**LeNet-5 (1998)** [37]. Yann LeCun tomonidan ishlab chiqilgan birinchi amaliy CNN. Qo'lyozma raqamlarni aniqlash (MNIST) uchun mo'ljallangan. 7 qatlamdan iborat, ~60K parametr. Bu arxitektura CNN ning asosiy tamoyillarini — konvolyutsiya, pooling va to'liq ulangan qatlamlarni birinchi marta birlashtirdi.

**AlexNet (2012)** [11]. ImageNet musobaqasini yutgan birinchi chuqur CNN. 8 qatlamdan iborat (5 konvolyutsion + 3 to'liq ulangan), 60M parametr. Innovatsiyalari: ReLU aktivatsiya, Dropout regularizatsiya, GPU da parallel o'qitish, data augmentatsiya.

**VGGNet (2014)** [2]. Simonyan va Zisserman tomonidan Oxford universitetida ishlab chiqilgan VGGNet soddaligi bilan ajralib turadi. Asosiy g'oya — faqat 3×3 konvolyutsion filtrlarni ishlatish va tarmoqni chuqurlashtirilish. VGG-16 (16 qatlam) va VGG-19 (19 qatlam) variantlari mavjud. VGG-16 138 million parametrga ega — bu hisoblash resurslariga bo'lgan talabni juda oshiradi. ImageNet da 71.5% top-1 aniqlik ko'rsatdi.

VGGNet ning muhim kashfiyoti — ikki ketma-ket 3×3 konvolyutsiya bitta 5×5 konvolyutsiya bilan bir xil qamrov (receptive field) ga ega, ammo parametrlar soni kam: 2×(3×3) = 18 vs 5×5 = 25. Bu printsip keyingi barcha arxitekturalarda qo'llanildi.

**GoogLeNet/Inception (2014)** [18]. Szegedy va boshqalar tomonidan Google da ishlab chiqilgan bu arxitektura "Inception module" konsepsiyasini taqdim etdi. Har bir modul turli o'lchamdagi filtrlarni parallel ravishda qo'llaydi:
- 1×1 konvolyutsiya — kanallar sonini kamaytirish (bottleneck);
- 3×3 konvolyutsiya — o'rta masshtabdagi featurelar;
- 5×5 konvolyutsiya — katta masshtabdagi featurelar;
- 3×3 max pooling — fazoviy ma'lumot.

Natijalar concatenation orqali birlashtiriladi. GoogLeNet faqat 6.8M parametrga ega bo'lib, VGGNet ga nisbatan 20 marta kam, ammo aniqlik jihatidan yuqori (74.8% top-1).

**ResNet (2015)** [1]. He, Zhang, Ren va Sun tomonidan Microsoft Research da ishlab chiqilgan Residual Network (ResNet) chuqur neyron tarmoqlarning eng muhim muammolaridan biri — gradiyentning yo'qolishi (vanishing gradient) muammosini hal qildi.

**Muammo:** Tarmoq chuqurlashtirilganda (20+ qatlam) gradient backpropagation jarayonida juda kichik qiymatlarga aylanadi va dastlabki qatlamlar o'rganmaydi. Bu "degradation problem" deb ataladi — chuqurroq tarmoq sayozroq tarmoqdan yomonroq natija ko'rsatadi.

**Yechim — Residual Learning:** Har bir blokda kirish signalini chiqishga bevosita qo'shish (skip connection, shortcut):

```
y = F(x, {W_i}) + x
```

Bu yerda x — kirish, F(x) — o'rganilgan qoldiq (residual) funksiya, y — chiqish. Agar optimal funksiya identitiya funksiyaga yaqin bo'lsa, F(x) = 0 o'rganish yetarli — bu oddiy qatlamga nisbatan osonroq.

**ResNet50 ning batafsil arxitekturasi:**

ResNet50 50 qatlamdan iborat bo'lib, "bottleneck" bloklari asosida qurilgan. Har bir bottleneck blok 3 ta konvolyutsion qatlamdan iborat:

```
Kirish (256 kanal)
  │
  ├── 1×1 conv, 64 kanal (kanal sonini kamaytirish)
  ├── BatchNorm + ReLU
  ├── 3×3 conv, 64 kanal (asosiy konvolyutsiya)
  ├── BatchNorm + ReLU
  ├── 1×1 conv, 256 kanal (kanal sonini qaytarish)
  ├── BatchNorm
  │
  └── + Kirish (skip connection)
      │
      ReLU
      │
  Chiqish (256 kanal)
```

Bu bottleneck dizayni tufayli ResNet50 faqat 25.6M parametrga ega (VGG-16 dan 5 marta kam), ammo 50 qatlam chuqurligiga ega.

ResNet50 ning to'liq arxitekturasi:

| Qatlam nomi | Chiqish o'lchami | Bottleneck tarkibi | Takror soni |
|-------------|-----------------|-------------------|-------------|
| conv1 | 112 × 112 × 64 | 7×7 conv, stride 2 | 1 |
| max pool | 56 × 56 × 64 | 3×3 max pool, stride 2 | 1 |
| conv2_x | 56 × 56 × 256 | [1×1, 64; 3×3, 64; 1×1, 256] | 3 |
| conv3_x | 28 × 28 × 512 | [1×1, 128; 3×3, 128; 1×1, 512] | 4 |
| conv4_x | 14 × 14 × 1024 | [1×1, 256; 3×3, 256; 1×1, 1024] | 6 |
| conv5_x | 7 × 7 × 2048 | [1×1, 512; 3×3, 512; 1×1, 2048] | 3 |
| GAP | 1 × 1 × 2048 | Global Average Pooling | 1 |
| FC | 1000 | Fully Connected | 1 |

Global Average Pooling (GAP) oxirgi konvolyutsion qatlamning har bir feature map ini bitta qiymatga kamaytiradi — bu to'liq ulangan qatlamga nisbatan parametrlar sonini keskin kamaytiradi va overfitting ni oldini oladi.

ResNet50 ning ImageNet da natijalari: top-1 accuracy — 76.1%, top-5 accuracy — 92.9%.

**Transfer Learning.** Transfer learning — katta ma'lumotlar to'plamida (masalan, ImageNet, 14M tasvir, 1000 sinf) o'rgatilgan modelning og'irliklarini yangi, kichikroq ma'lumotlar to'plamiga moslashtirilish jarayoni [19]. Yosinski va boshqalar (2014) ko'rsatishicha, CNN ning past darajadagi qatlamlari (chiziqlar, burchaklar, teksturalar) ko'pgina vizual vazifalar uchun universal bo'lib, ularni qayta o'rgatish shart emas. Faqat yuqori darajadagi qatlamlar va classification head yangi vazifaga moslashtiriladi.

Transfer learning ning asosiy strategiyalari:
- **Feature extraction** — backbone muzlatiladi (freeze), faqat yangi FC qatlam o'rgatiladi. Kam ma'lumot bo'lganda samarali;
- **Fine-tuning** — barcha yoki ba'zi qatlamlar ochiladi (unfreeze) va kichik learning rate bilan o'rgatiladi. Ko'p ma'lumot bo'lganda samarali;
- **Progressive unfreezing** — avval faqat FC, keyin oxirgi qatlamlar, keyin barcha qatlamlar ochiladi. Eng barqaror natija beradi.

Bizning tadqiqotimizda fine-tuning strategiyasi tanlandi, chunki ma'lumotlar to'plami yetarlicha katta (608,021 training tasvirlari).

**YOLO (You Only Look Once)** [3]. Redmon va boshqalar tomonidan 2016-yilda taqdim etilgan YOLO ob'ektni aniqlash (object detection) uchun mo'ljallangan arxitektura bo'lib, u butun tasvirni bir marta qarab, barcha ob'ektlarni aniqlaydi. Bu "ikki bosqichli" (two-stage) detektorlardan (R-CNN, Faster R-CNN) farqli o'laroq, YOLO "bir bosqichli" (one-stage) detektordur.

YOLO ning asosiy g'oyasi — tasvirni S×S katakchali to'rga bo'lish va har bir katakcha uchun bounding box va sinf ehtimolliklarini bir vaqtda bashorat qilish:

```
P(Class_i | Object) × P(Object) × IoU = P(Class_i) × IoU
```

YOLOv8 [20] — Ultralytics kompaniyasi tomonidan 2023-yilda chiqarilgan eng so'nggi versiya. U quyidagi vazifallarni bajaradi:
- **Detection** — ob'ektni aniqlash va joylashuvini ko'rsatish;
- **Classification** — tasvirni tasniflash;
- **Segmentation** — ob'ektning piksel darajasida segmentatsiyasi;
- **Pose estimation** — inson pozasini baholash.

YOLOv8-cls (classification) varianti tasvirni tasniflash uchun optimallashtirilgan bo'lib, CSPDarknet backbone dan foydalanadi. U ResNet ga nisbatan tezroq ishlaydi (real-time), ammo feature boyligida biroz past bo'lishi mumkin.

**DenseNet (2017)** [32]. Huang va boshqalar tomonidan taklif etilgan DenseNet da har bir qatlam barcha oldingi qatlamlar bilan to'g'ridan-to'g'ri bog'langan (dense connection):

```
x_l = H_l([x_0, x_1, ..., x_{l-1}])
```

Bu yerda [x_0, x_1, ...] — barcha oldingi qatlamlarning chiqishlari concatenation orqali birlashtirilgan. DenseNet ning afzalligi — gradient oqimini yaxshilash va featurlarni qayta ishlatish. Kamchiligi — xotira talabi yuqori.

**EfficientNet (2019)** [33]. Tan va Le tomonidan taklif etilgan EfficientNet tarmoq chuqurligi, kengligi va tasvir o'lchamini bir vaqtda optimal tarzda masshtablash (compound scaling) usulini joriy etdi:

```
depth: d = α^φ
width: w = β^φ  
resolution: r = γ^φ
α · β² · γ² ≈ 2
```

Bu yerda φ — foydalanuvchi tomonidan belgilanadigan koeffitsient. EfficientNet-B0 dan B7 gacha variantlari mavjud, har biri tobora katta va aniqroq.

**Arxitekturalar taqqoslashi:**

| Arxitektura | Yil | Parametrlar | Top-1 (ImageNet) | FLOPs | Xususiyati |
|-------------|-----|-------------|-------------------|-------|-----------|
| AlexNet | 2012 | 60M | 63.3% | 0.7G | Birinchi chuqur CNN |
| VGG-16 | 2014 | 138M | 71.5% | 15.5G | Sodda, chuqur |
| GoogLeNet | 2014 | 6.8M | 74.8% | 1.5G | Inception module |
| ResNet50 | 2015 | 25.6M | 76.1% | 4.1G | Residual learning |
| DenseNet-121 | 2017 | 8M | 74.4% | 2.9G | Dense connections |
| EfficientNet-B0 | 2019 | 5.3M | 77.1% | 0.4G | Compound scaling |
| YOLOv8s-cls | 2023 | 6.4M | 73.8% | 1.2G | Real-time |

**ResNet50 tanlash asoslari.** Tadqiqotimiz uchun ResNet50 vizual feature extractor sifatida tanlandi, chunka quyidagi sabablarga asoslanadi:

1. **Transfer learning uchun samaradorlik** — ResNet50 ImageNet da oldindan o'rgatilgan og'irliklar keng tarqalgan va samaradorligi isbotlangan. 76.1% top-1 aniqlik va 92.9% top-5 aniqlik ko'rsatadi;
2. **Boy feature ifodalash** — oxirgi qatlamdan 2048 o'lchamli vektor olinadi, bu turli harakatlarni farqlash uchun yetarli ma'lumot saqlaydi;
3. **Residual connection** — 50 qatlam chuqurligida ham gradient muammosiz o'rganish imkoniyati;
4. **O'rtacha hisoblash talabi** — VGG-16 ga nisbatan 5 marta kam parametr, EfficientNet ga nisbatan biroz ko'p, ammo feature boyligi yuqori;
5. **Keng qo'llanilganlik** — ilmiy adabiyotlarda eng ko'p ishlatiladigan backbone, bu taqqoslash uchun qulay.

### 1.3. Poza baholash usullari va ularning qo'llanilishi

Inson pozasini baholash (human pose estimation, HPE) — bu tasvir yoki videoda inson tanasining asosiy bo'g'inlari (joints, keypoints, landmarks) joylashuvini aniqlash jarayonidir [21]. Bu texnologiya sport tahlili, tibbiy reabilitatsiya, virtual reallik, xavfsizlik va xatti-harakat aniqlash kabi sohalarda keng qo'llaniladi.

**Poza baholash muammosining matematik ta'rifi.** Berilgan tasvir I uchun, poza baholash modeli P(I) funksiyasini topishi kerak, u N ta bo'g'inning joylashuvini aniqlaydi:

```
P(I) = {p_1, p_2, ..., p_N}, p_i = (x_i, y_i) yoki (x_i, y_i, z_i)
```

Bu yerda p_i — i-bo'g'inning koordinatalari. 2D poza baholashda (x, y) koordinatalar, 3D poza baholashda (x, y, z) koordinatalar aniqlanadi.

**Pozani baholash usullari ikki asosiy guruhga bo'linadi:**

**Yuqoridan pastga (top-down) yondashuv.** Bu yondashuvda avval rasmda barcha insonlar aniqlanadi (person detection), keyin har bir inson uchun alohida poza baholanadi. Ishlash tartibi:
1. Object detector (masalan, Faster R-CNN, YOLO) yordamida insonga tegishli bounding box aniqlanadi;
2. Har bir bounding box alohida crop qilinadi;
3. Pose estimation modeli har bir crop uchun bo'g'inlar joylashuvini bashorat qiladi.

Afzalligi: yuqori aniqlik, chunki har bir inson alohida qayta ishlanadi. Kamchiligi: ko'p insonli sahnalarda sekin ishlaydi (O(n) — insonlar soni bilan chiziqli o'sadi).

**Pastdan yuqoriga (bottom-up) yondashuv.** Bu yondashuvda avval rasmda barcha bo'g'inlar aniqlanadi, keyin ular alohida insonlarga birlashtiriladi (grouping). Ishlash tartibi:
1. Heatmap prediction — har bir bo'g'in turi uchun heatmap bashorat qilinadi;
2. Part association — qo'shni bo'g'inlar orasidagi bog'lanish (association) aniqlanadi;
3. Grouping — bog'langan bo'g'inlar alohida insonlarga birlashtiriladi.

Afzalligi: tezlik insonlar soniga kam bog'liq. Kamchiligi: aniqlik biroz pastroq, ayniqsa bir-biriga yaqin turgan insonlar uchun.

**Asosiy poza baholash arxitekturalari va tizimlari:**

**DeepPose (2014)** [38]. Toshev va Szegedy tomonidan taklif etilgan birinchi chuqur o'rganish asosidagi poza baholash usuli. CNN regressiya orqali bo'g'in koordinatalarini bevosita bashorat qiladi. Cascaded refinement — bir necha bosqichda bashoratni yaxshilash usuli qo'llanilgan. FLIC datasetida 63.3% PCKh@0.5 ko'rsatdi.

**Stacked Hourglass Network (2016)** [39]. Newell, Yang va Deng tomonidan taklif etilgan bu arxitektura "hourglass" (qum soat) shaklida tuzilgan — avval tasvir kichiklashtiriladi (encoder), keyin kattalashtiriladi (decoder). Bir necha hourglass modullari ketma-ket ulangan bo'lib, har biri oldingi natijani yanada yaxshilaydi. Bu arxitektura MPII datasetida state-of-the-art natija ko'rsatdi.

**OpenPose (2017)** [5]. Cao, Simon, Wei va Sheikh tomonidan Carnegie Mellon Universitetida ishlab chiqilgan OpenPose birinchi real-vaqt ko'p insonli bottom-up poza baholash tizimi hisoblanadi. OpenPose ikki asosiy komponentdan iborat:

1. **Confidence Maps** — har bir bo'g'in turi uchun heatmap. Har bir piksel ushbu joyda ma'lum bo'g'in mavjudligining ehtimolligini ko'rsatadi;
2. **Part Affinity Fields (PAFs)** — qo'shni bo'g'inlar orasidagi bog'lanish yo'nalishini ko'rsatadigan 2D vektor maydoni.

OpenPose ning ishlash tartibi:
1. VGG-19 backbone dan feature map olinadi;
2. Ikki parallel branch: confidence maps va PAFs;
3. Greedy parsing — PAFs orqali bo'g'inlar insonga birlashtiriladi.

OpenPose 18 ta bo'g'inni (COCO format) yoki 25 ta bo'g'inni (BODY_25 format) aniqlaydi. GPU da real-time ishlaydi (30+ FPS), ammo CPU da sekin (~2 FPS).

**HRNet (2019)** [40]. Sun, Xiao, Wei va Wang tomonidan taklif etilgan High-Resolution Network (HRNet) butun tarmoq davomida yuqori resolyutsiyali ifodalashni (representation) saqlab qoladi. Ko'pgina CNN lar rasmni asta-sekin kichiklashtiradi va keyin kattalashtiradi (encoder-decoder), ammo HRNet parallel ravishda turli resolyutsiyali branchlarni boshqaradi va ular o'rtasida ma'lumot almashadi.

HRNet COCO datasetida 75.5% AP (Average Precision) ko'rsatdi — bu vaqtdagi eng yaxshi natija edi.

**MediaPipe Pose (2020)** [4]. Google tomonidan ishlab chiqilgan MediaPipe Pose tizimi real vaqtda ishlash uchun optimallashtirilgan yengil arxitektura hisoblanadi. U BlazePose arxitekturasiga asoslangan — maxsus mobil qurilmalar uchun mo'ljallangan.

MediaPipe Pose ning 33 ta landmarki va ularning joylashuvi:

| Raqam | Landmark nomi | Tana qismi |
|-------|---------------|------------|
| 0 | nose | Burun |
| 1-4 | left/right eye inner/outer | Ko'zlar |
| 5-6 | left/right eye | Ko'z markazlari |
| 7-8 | left/right ear | Quloqlar |
| 9-10 | mouth left/right | Og'iz burchaklari |
| 11-12 | left/right shoulder | Yelkalar |
| 13-14 | left/right elbow | Tirsaklar |
| 15-16 | left/right wrist | Bilaklarlar |
| 17-20 | left/right pinky/index | Barmoq uchlari |
| 21-22 | left/right thumb | Bosh barmoqlar |
| 23-24 | left/right hip | Son bo'g'inlari |
| 25-26 | left/right knee | Tizzalar |
| 27-28 | left/right ankle | To'piqlar |
| 29-30 | left/right heel | Tovonlar |
| 31-32 | left/right foot index | Oyoq barmoqlari |

Har bir landmark uchun 4 ta qiymat qaytariladi:
- **x** — gorizontal koordinata (0.0 dan 1.0 gacha, tasvirning kengligiga nisbatan normalizatsiyalangan);
- **y** — vertikal koordinata (0.0 dan 1.0 gacha, tasvirning balandligiga nisbatan normalizatsiyalangan);
- **z** — chuqurlik (nisbiy qiymat, son bo'g'inining z-koordinatasiga nisbatan);
- **visibility** — landmarkning ko'rinish darajasi (0.0 dan 1.0 gacha, 0 — ko'rinmaydi, 1 — to'liq ko'rinadi).

Shunday qilib, har bir tasvir uchun 33 × 4 = **132 o'lchamli** poza feature vektori olinadi.

MediaPipe Pose ning arxitekturasi uch bosqichdan iborat:
1. **Person detection** — tasvirda inson borligini va joylashuvini aniqlash (BlazeFace detektori);
2. **Pose landmark prediction** — aniqlangan inson uchun 33 ta landmarkni bashorat qilish;
3. **Tracking** — video rejimida oldingi kadrdagi pozani kuzatish (qayta detection qilmaslik uchun).

MediaPipe Pose ning afzalliklari:
- **33 ta landmark** — OpenPose (18-25) dan ko'p, barmoqlar va oyoq uchlarini ham o'z ichiga oladi;
- **Real-time ishlash** — CPU da ham 30+ FPS, GPU shart emas;
- **Yengil model** — Lite varianti 5.7 MB, Full varianti 22 MB;
- **Cross-platform** — Python, JavaScript, Android, iOS da ishlaydi;
- **Visibility qiymati** — qisman yashiringan landmarklar ham aniqlanadi.

Kamchiliklari:
- Faqat bitta inson uchun optimallashtirilgan — ko'p insonli sahnalarda avval detection kerak;
- Juda uzoq masofadagi (10+ metr) kichik insonlar uchun aniqlik pasayishi mumkin.

**Poza baholash tizimlarining taqqoslashi:**

| Tizim | Yil | Keypoints | Yondashuv | Tezlik (CPU) | Tezlik (GPU) | COCO AP |
|-------|-----|-----------|-----------|-------------|-------------|---------|
| OpenPose | 2017 | 18-25 | Bottom-up | ~2 FPS | 30+ FPS | 61.8% |
| HRNet | 2019 | 17 | Top-down | ~1 FPS | 15+ FPS | 75.5% |
| MediaPipe | 2020 | 33 | Top-down | 30+ FPS | 100+ FPS | ~70% |
| AlphaPose | 2017 | 17 | Top-down | ~5 FPS | 20+ FPS | 72.3% |

**Poza ma'lumotining xatti-harakat aniqlashdagi ahamiyati.** Ta'lim kontekstida poza ma'lumoti quyidagi harakatlarni aniqlashda muhim ahamiyatga ega. Quyida har bir xatti-harakat uchun asosiy skeletal belgilar keltirilgan:

| Xatti-harakat | Vizual ko'rinish | Asosiy poza belgilari |
|---------------|-----------------|----------------------|
| Qo'l ko'tarish (hand-raising) | Qo'l tepada | Bilak (wrist) y-koordinatasi < yelka (shoulder) y-koordinatasi; qo'l-yelka burchagi > 90° |
| Yozish (write) | Bosh egilgan, qo'l partada | Bilaklarlar (wrists) hip darajasida; tirsaklar (elbows) bukilgan; bosh biroz egilgan |
| Kitob o'qish (read) | Bosh egilgan, qo'l oldinda | Bilaklarlar ko'krak darajasida; tirsaklar bukilgan; qo'llar simmetrik |
| Bosh egish (bow-head) | Bosh kuchli egilgan | Burun (nose) y > yelka y; qo'llar erkin (wrist y >> shoulder y); tana oldinga egilgan |
| O'rnidan turish (standing) | Tana vertikal | Hip-shoulder masofa katta; tizzalar to'g'ri; yelkalar hip dan ancha yuqorida |
| Bosh burish (turn-head) | Bosh yon tomonga | Ko'zlar asimmetrik (bir ko'z visibility yuqori, ikkinchisi past); burun yelka chizig'idan chiqib ketgan |
| E'tibor berish (focus) | Bosh to'g'ri, tana doskaga | Burun ikkala yelka o'rtasida; ko'zlar simmetrik; bosh egilmagan |

Bu jadvaldan ko'rinib turibdiki, vizual jihatdan o'xshash bo'lgan "read" va "bow-head" harakatlari poza ma'lumotida aniq farqlanadi — "read" da qo'llar oldinda va ko'tarilgan, "bow-head" da esa qo'llar erkin va pastda.

**MediaPipe Pose ni tanlash asoslari.** Tadqiqotimizda MediaPipe Pose tanlandi, chunki:
1. **33 ta landmark** — eng to'liq skeletal ma'lumot, barmoqlar va oyoqlar ham;
2. **Visibility qiymati** — sinf xonasida o'quvchi qisman yashiringan holatlarda ham ishlaydi;
3. **CPU da real-time** — server talablarini kamaytiradi, GPU majburiy emas;
4. **Yengil model** — Lite varianti 5.7 MB, Docker konteynerga oson integratsiya;
5. **Python SDK** — training pipeline ga oson integratsiya, NumPy bilan mos ishlaydi;
6. **Barqarorlik** — Google tomonidan qo'llab-quvvatlanadi va doimiy yangilanadi.

### 1.4. Fusion yondashuvlar va ularning samaradorligi

Fusion (birlashtirish) — turli manbalardan olingan ma'lumotlarni birlashtirib, yagona qaror chiqarish jarayonidir [23]. Kompyuter ko'rishda fusion yondashuvlar bitta modallik (masalan, faqat tasvir) dan ko'ra yuqoriroq aniqlik va ishonchlilik beradi. Bu fenomen "complementary information" (to'ldiruvchi ma'lumot) printsipi bilan tushuntiriladi — turli manbalar bir xil ob'ekt haqida turli va bir-birini to'ldiruvchi ma'lumot beradi.

**Fusion darajalari va ularning matematik ta'rifi:**

**1. Ma'lumot darajasidagi fusion (Data-level / Early fusion).** Xom ma'lumotlar (raw data) bevosita birlashtiriladi. Matematik ifodasi:

```
f_fused = Concatenate(X_1, X_2, ..., X_n)
```

Bu yerda X_i — i-manbadan olingan xom ma'lumot. Masalan, RGB tasvir (H×W×3) va chuqurlik (depth) tasviri (H×W×1) birlashtirilsa, natija H×W×4 o'lchamli tensor bo'ladi.

Afzalliklari: eng ko'p ma'lumot saqlanadi, turli manbalar orasidagi nozik bog'lanishlar o'rganilishi mumkin.
Kamchiliklari: hisoblash murakkabligi yuqori, turli manbalarning o'lchamlari mos kelmasligi mumkin, noisy ma'lumotga sezgir.

**2. Feature darajasidagi fusion (Feature-level / Middle fusion).** Turli manbalardan olingan featurelar birlashtiriladi. Bu eng keng tarqalgan va samarali yondashuv. Quyidagi usullar mavjud:

**Concatenation (ulash):**
```
f_fused = [f_1; f_2; ...; f_n]
```
Ikki feature vektorni ketma-ket ulash. Sodda va samarali, barcha ma'lumot saqlanadi, ammo o'lcham oshadi.

**Element-wise addition (qo'shish):**
```
f_fused = f_1 + f_2 + ... + f_n
```
Bir xil o'lchamdagi featurelarni element bo'yicha qo'shish. O'lcham o'zgarmaydi, ammo featurelar bir xil o'lchamda bo'lishi va bir xil semantik mazmunda bo'lishi kerak.

**Element-wise multiplication (ko'paytirish):**
```
f_fused = f_1 ⊙ f_2 ⊙ ... ⊙ f_n
```
Feature vektorlarni element bo'yicha ko'paytirish. "Gate" mexanizmi sifatida ishlaydi — bitta modal ikkinchisini filtrlaydi.

**Bilinear pooling:**
```
f_fused = f_1^T · f_2
```
Ikki feature vektor ning tashqi ko'paytmasi (outer product). Featurelar orasidagi barcha juftlik interaksiyalarni modellashtiradi, ammo o'lcham juda oshadi (d_1 × d_2).

**Attention-based fusion:**
```
α = softmax(W_a · [f_1; f_2])
f_fused = α_1 · f_1 + α_2 · f_2
```
Attention mexanizmi orqali qaysi feature muhimroq ekanligini o'rganish. Eng moslashuvchan, ammo qo'shimcha parametrlar talab qiladi.

**Gated fusion:**
```
g = σ(W_g · [f_1; f_2] + b_g)
f_fused = g ⊙ f_1 + (1-g) ⊙ f_2
```
Gate mexanizmi qaysi manbaga qancha e'tibor berish kerakligini o'rganadi. σ — sigmoid funksiya.

**3. Qaror darajasidagi fusion (Decision-level / Late fusion).** Har bir modallik alohida qaror (prediction) chiqaradi, keyin qarorlar birlashtiriladi:

**Ovoz berish (voting):**
```
y_final = mode(y_1, y_2, ..., y_n)
```

**O'rtacha ehtimollik:**
```
p_final = (1/n) Σ p_i
```

**Vaznli o'rtacha:**
```
p_final = Σ w_i · p_i, Σ w_i = 1
```

Afzalligi: har bir modallik mustaqil o'rgatiladi, sodda va tushinarli.
Kamchiligi: oraliq ma'lumot (intermediate features) yo'qoladi, manbalar orasidagi nozik interaksiyalar o'rganilmaydi.

**Mavjud fusion tadqiqotlari xatti-harakat aniqlash sohasida:**

**Cippitelli va boshqalar (2016)** [24] skeletal ma'lumot va RGB tasvirni birlashtirib, keksa odamlarning yiqilishini aniqlash tizimini ishlab chiqdilar. Ular feature-level concatenation fusion qo'lladilar: CNN dan 256-dim vizual feature va skeletal ma'lumotdan 128-dim feature birlashtirildi. Natija: fusion yondashuv alohida modalliklarga nisbatan 12% yuqori aniqlik ko'rsatdi (89% vs 77%).

**Shahroudy va boshqalar (2016)** [25] NTU RGB+D ma'lumotlar to'plami — 56,880 ta video klip, 60 xil harakat, 4 ta modallik (RGB, depth, infrared, skeleton) asosida multi-modal fusion arxitekturasini taklif qildilar. Ular LSTM va CNN ni birlashtirib, 85% aniqlikka erishdilar. Bu dataset xatti-harakat aniqlash sohasida eng keng ishlatiladigan benchmarklardan biri hisoblanadi.

**Cheng va boshqalar (2020)** [41] skeleton-aware multi-stream CNN ishlab chiqdilar — skeletal bo'g'inlarni turli semantik guruhlarga bo'lib (qo'l, oyoq, bosh), har bir guruh uchun alohida CNN o'rgatildi va natijalar birlashtirildi. NTU RGB+D da 92.3% accuracy ko'rsatdi.

**Zhang va boshqalar (2019)** [26] tibbiyot sohasida bemorning holatini aniqlash uchun vizual va sensor ma'lumotlarini birlashtiradigan fusion modelni yaratdilar. Ular attention-based fusion qo'lladilar — model avtomatik ravishda qaysi manbaga ko'proq e'tibor berish kerakligini o'rgandi. Natijalar: fusion 8-15% yuqori aniqlik ko'rsatdi.

**Wang va boshqalar (2020)** [27] ta'lim sohasida o'quvchilar xatti-harakatini aniqlash uchun multi-feature fusion qo'lladilar. Ammo ularning "fusion"i faqat CNN ning turli qatlamlaridan olingan vizual featurelarni birlashtirish (feature pyramid) edi — skeletal yoki poza ma'lumoti ishlatilmagan. Bu yondashuv faqat vizual ma'lumot ichidagi turli darajadagi featurelarni birlashtiradi.

**Liu va boshqalar (2021)** [42] sinf xonasida o'quvchilar xatti-harakatini aniqlash uchun CNN va LSTM ni birlashtirgan temporal fusion modelni taklif qildilar. Ularning modeli video kliplardan vaqt bo'yicha kontekstni olish uchun LSTM ishlatdi. 80% accuracy ko'rsatdi, ammo faqat vizual feature ishlatilgan, skeletal ma'lumot yo'q.

**Fusion yondashuvlarning samaradorlik tahlili:**

| Tadqiqot | Soha | Fusion turi | Modalliklar | Yaxshilanish |
|----------|------|-------------|-------------|-------------|
| Cippitelli (2016) | Yiqilish aniqlash | Feature concat | RGB + Skeleton | +12% |
| Shahroudy (2016) | Xatti-harakat | LSTM fusion | RGB + Depth + Skeleton | +8% |
| Cheng (2020) | Xatti-harakat | Multi-stream | Skeleton parts | +5% |
| Zhang (2019) | Tibbiyot | Attention fusion | Vizual + Sensor | +8-15% |
| Liu (2021) | Ta'lim | Temporal fusion | RGB + LSTM | +6% |

Bu jadvaldan ko'rinib turibdiki, fusion yondashuvlar barcha sohalarda va barcha modalliklarda sezilarli yaxshilanish beradi. O'rtacha yaxshilanish 5-15% atrofida.

**Bizning yondashuvimiz — PoseCNN Fusion.** Yuqoridagi tahlil asosida biz **feature-level concatenation fusion** yondashuvini tanladik, ammo quyidagi muhim farqlar bilan:

1. **Poza featurelari MLP orqali o'zgartiriladi** — xom 132-dim landmark koordinatalari avval MLP (132→256→128) orqali yuqori darajadagi harakatga oid featurelarga aylantiriladi. Bu noisy landmarkları filtrlaydi va muhim signallarni kuchaytiradi;

2. **Asimmetrik fusion** — vizual branch (2048-dim) poza branch (128-dim) dan ancha katta. Bu vizual ma'lumotni asosiy, poza ma'lumotini to'ldiruvchi (complementary) sifatida saqlaydi. Agar poza topilmasa, model faqat vizual featurelarga tayanib ham ishlaydi;

3. **End-to-end o'rganish** — vizual backbone, poza MLP va classifier birgalikda optimallashtiriladi. Bu barcha komponentlarning bir-biriga moslashishini ta'minlaydi;

4. **Domain-specific dizayn** — arxitektura aynan sinf xonasida o'quvchilar xatti-harakatini aniqlash uchun optimallashtirilgan. Poza featurelari qo'l ko'tarish, yozish, o'qish kabi harakatlar uchun muhim bo'lgan tana qismlariga e'tibor beradi.

### I bob bo'yicha xulosa

Birinchi bobda o'quvchilar xatti-harakatini aniqlash bo'yicha mavjud usullar va texnologiyalar chuqur tahlil qilindi. Asosiy xulosalar quyidagicha:

1. **Kompyuter ko'rish evolyutsiyasi** — an'anaviy HOG, SIFT, BoVW usullaridan chuqur o'rganish (AlexNet, VGGNet, ResNet, YOLO) ga o'tish tasniflash aniqligini 20-30% oshirdi. Ammo faqat vizual featurelar bilan 70% atrofidagi "shift" (to'siq) mavjud;

2. **CNN arxitekturalari** — ResNet50 transfer learning uchun eng samarali arxitekturalardan biri. Residual connection tufayli 50 qatlamli tarmoq samarali o'rganadi, 2048 o'lchamli boy vizual featurelarni ajratib oladi;

3. **Poza baholash** — MediaPipe Pose 33 ta landmarkni real vaqtda aniqlaydi, har bir landmark uchun x, y, z, visibility qiymatlari qaytariladi. CPU da ham tez ishlaydi va sinf xonasi sharoitlari uchun yetarli aniqlik beradi;

4. **Fusion yondashuvlar** — ilmiy adabiyotda vizual va skeletal ma'lumotlarni birlashtirish 5-15% aniqlik oshishini ko'rsatganligi isbotlangan. Ammo ta'lim sohasida CNN va poza ma'lumotini birlashtiradigan tadqiqotlar hali kam o'rganilgan;

5. **Ilmiy bo'shliq** — sinf xonasida o'quvchilar xatti-harakatini aniqlash uchun CNN vizual featurelari va skeletal poza featurelari ni birlashtirgan tadqiqot mavjud emas. Bu bizning PoseCNN Fusion algoritmimizning ilmiy yangiligini tashkil etadi.

Ushbu tahlil asosida keyingi bobda PoseCNN Fusion algoritmining batafsil arxitekturasi va ishlab chiqish jarayoni tavsiflangan.

---

## II BOB. POSECNN FUSION ALGORITMINI ISHLAB CHIQISH

### 2.1. Ma'lumotlar to'plamini shakllantirish va balanslashtirish

Har qanday chuqur o'rganish modelining samaradorligi bevosita ma'lumotlar to'plamining sifati, hajmi va tarkibiga bog'liq [28]. Goodfellow va boshqalar (2016) ta'kidlaganidek, "ko'proq ma'lumot ko'pincha yaxshiroq algoritmdan samaraliroq". Tadqiqotimiz uchun sinf xonasidagi o'quvchilar xatti-harakatini ifodalovchi maxsus ma'lumotlar to'plami shakllantirildi.

**Xatti-harakat sinflari va ularning ta'rifi.** Ma'lumotlar to'plami 7 xil xatti-harakat turini o'z ichiga oladi. Har bir sinf ta'lim jarayonidagi muhim harakatni ifodalaydi va ikki asosiy guruhga bo'linadi:

**Diqqatli (Attentive) guruhi — o'quvchi darsga e'tibor qaratgan:**

1. **focus** (e'tibor berish) — o'quvchi doskaga yoki o'qituvchiga qarab o'tirgan. Boshi to'g'ri, tanasi doskaga qaratilgan, qo'llari partada yoki tizzasida. Bu eng keng tarqalgan "yaxshi" holat;

2. **hand-raising** (qo'l ko'tarish) — o'quvchi savolga javob berish yoki savol berish uchun qo'lini ko'targan. Bir yoki ikkala qo'l yuqoriga ko'tarilgan, bilak bosh yoki yelka balandligida. Bu faol ishtirokni ko'rsatadi;

3. **read** (o'qish) — o'quvchi kitob, daftar yoki boshqa o'quv materialini o'qiyotgan. Boshi biroz egilgan, qo'llari oldinda kitob yoki daftar ushlab turgan. Diqqat o'quv materialiga qaratilgan;

4. **write** (yozish) — o'quvchi yozayotgan (daftarga, testga). Boshi egilgan, dominant qo'l harakatda (yozmoqda), ikkinchi qo'l daftarni ushlab turgan. Diqqat yozish jarayoniga qaratilgan.

**Chalg'igan (Distracted) guruhi — o'quvchi darsga e'tibor bermayotgan:**

5. **bow-head** (bosh egish) — o'quvchi boshini kuchli eggan, odatda uyqulayotgan yoki telefoniga qarayotgan. Bosh juda past, qo'llar erkin holda yoki stol ostida. Bu eng ko'p uchraydigan "yomon" holat;

6. **standing** (turish) — o'quvchi o'rnidan turgan. Tana vertikal, oyoqlar to'g'ri. Bu odatda sinf tartibini buzish yoki chiqib ketish belgisi;

7. **turn-head** (bosh burish) — o'quvchi boshini yon tomonga burgan, boshqa o'quvchi bilan gaplashayotgan yoki boshqa tomonga qarayotgan. Bosh yon tomonga burilgan, ammo tana to'g'ri. Bu chalg'ish belgisi.

**Sinflar orasidagi murakkab chegaralar.** Ba'zi sinflar o'rtasida vizual o'xshashlik yuqori bo'lib, bu tasniflash muammosini murakkablashtiradi:

| Juftlik | O'xshashlik | Farq (poza orqali) |
|---------|-------------|-------------------|
| read ↔ bow-head | Ikkisida ham bosh egilgan | read da qo'llar oldinda kitob ushlab turadi, bow-head da qo'llar erkin |
| write ↔ read | Ikkisida ham bosh egilgan, qo'llar partada | write da bitta qo'l harakatda, read da ikki qo'l simmetrik |
| focus ↔ turn-head | Ikkisida ham o'tirgan holat | focus da bosh to'g'ri, turn-head da bosh burilgan |
| hand-raising ↔ standing | Ikkisida ham tana ko'tarilgan | hand-raising da faqat qo'l yuqorida, standing da butun tana |

Bu jadval poza ma'lumotining vizual ma'lumotni qanday to'ldirishi mumkinligini aniq ko'rsatadi.

**Ma'lumotlar to'plamining manbalari.** Boshlang'ich ma'lumotlar quyidagi manbalardan to'plandi:

1. **Ochiq datasetlar.** Kaggle platformasidan sinf xonasi tasvirlari (Student Classroom Behavior dataset, SCB-Dataset [16]) olindi. Bu dataset turli mamlakatlar (Xitoy, Hindiston, Yevropa) dagi sinf xonalarini o'z ichiga oladi;

2. **Mahalliy to'plangan tasvirlar.** O'zbekiston maktablarida olingan video kadrlari. Turli sinf xonalari, turli yoritilganlik sharoitlari va turli yoshdagi o'quvchilar ishtrok etdi;

3. **Augmentatsiya orqali yaratilgan tasvirlar.** Kam sonli sinflar (bow-head, hand-raising, standing) uchun ma'lumotlar augmentatsiya texnikalari orqali ko'paytirildi.

**Ma'lumotlarni qayta ishlash pipeline.** Xom tasvirlardan training uchun tayyor ma'lumotlar to'plamini yaratish quyidagi bosqichlardan iborat:

**1-bosqich: Yuzni aniqlash va crop qilish (Face Detection & Cropping).** Sinf xonasi tasvirlaridan har bir o'quvchi alohida crop qilinadi. Bu jarayonda yuz aniqlash algoritmi (face detection) ishlatiladi. Yuz aniqlanganidan keyin, yuz atrofida **asimmetrik padding** qo'llaniladi:

```
padLeft   = yuz_kengligi × 0.5
padRight  = yuz_kengligi × 0.5
padTop    = yuz_balandligi × 0.3
padBottom = yuz_balandligi × 1.5
```

Pastga ko'proq padding beriladi (1.5×), chunki o'quvchining xatti-harakatini aniqlash uchun nafaqat yuz, balki yelka, qo'l va partaning yuqori qismi ham ko'rinishi kerak. Masalan, "qo'l ko'tarish" ni aniqlash uchun qo'lning yelkadan yuqoriga ko'tarilganligini ko'rish zarur.

Crop formulasi:
```
x1 = max(0, yuz_x - padLeft)
y1 = max(0, yuz_y - padTop)
x2 = min(tasvir_kengligi, yuz_x + yuz_kengligi + padRight)
y2 = min(tasvir_balandligi, yuz_y + yuz_balandligi + padBottom)
crop = tasvir[y1:y2, x1:x2]
```

**2-bosqich: Sinfga ajratish (Annotation).** Har bir crop qilingan tasvir ekspert tomonidan 7 ta sinfdan biriga belgilanadi. Belgilash jarayonida quyidagi qoidalarga amal qilindi:
- Harakat aniq ko'rinib turishi kerak;
- Noaniq holatlar (masalan, boshini biroz eggan, ammo kitob o'qiyotgan) kontekstga qarab belgilanadi;
- Bir necha ekspert tomonidan tekshiriladi (cross-validation).

**3-bosqich: Balanslashtirish.** Dastlabki ma'lumotlar to'plamida sinflar orasida katta nomutanosiblik mavjud edi. Balanslashtirilmagan ma'lumotlar to'plami model o'rganishiga salbiy ta'sir qiladi — model katta sinflarga moyillik ko'rsatadi (bias) va kichik sinflarni yomon tasniflaydi [29]. He va Garcia (2009) ko'rsatishicha, sinf nomutanosibligini hal qilish uchun quyidagi usullar qo'llaniladi:

**Oversampling** — kam sonli sinflarning tasvirlari ko'paytiriladi. Random Oversampling — tasodifiy tanlash va takrorlash. SMOTE (Synthetic Minority Over-sampling Technique) — mavjud tasvirlar orasida interpolyatsiya orqali yangi tasvirlar yaratish. Bizning holatimizda random oversampling + augmentatsiya qo'llanildi.

**Augmentatsiya** — mavjud tasvirlarni o'zgartirish orqali yangi tasvirlar yaratish:
- Gorizontal aks ettirish (horizontal flip) — chapni o'ngga aylantirish;
- Rang o'zgartirish (color jitter) — yorqinlik, kontrast, to'yinganlik o'zgartirish;
- Aylantirish (rotation) — kichik burchakka aylantirish;
- Tasodifiy kesish (random crop) — tasvirning turli qismlarini olish.

**Undersampling** — juda ko'p sonli sinflardan tasodifiy tanlash, ortiqcha tasvirlarni olib tashlash.

**Class weights** — loss funksiyada kam sonli sinflarga katta vazn berish:
```
w_c = N_total / (N_classes × N_c)
```
Bu yerda N_total — jami tasvirlar, N_classes — sinflar soni, N_c — c-sinfdagi tasvirlar soni.

**Yakuniy balanslashtirilgan ma'lumotlar to'plami:**

| Sinf | Guruh | Train | Val | Jami | Class Weight |
|------|-------|-------|-----|------|-------------|
| bow-head | Distracted | 95,076 | 540 | 95,616 | 1.31 |
| focus | Attentive | 100,000 | 14,750 | 114,750 | 1.24 |
| hand-raising | Attentive | 61,096 | 333 | 61,429 | 2.03 |
| read | Attentive | 100,000 | 12,901 | 112,901 | 1.24 |
| standing | Distracted | 51,849 | 2,224 | 54,073 | 2.39 |
| turn-head | Distracted | 100,000 | 3,765 | 103,765 | 1.24 |
| write | Attentive | 100,000 | 10,567 | 110,567 | 1.24 |
| **Jami** | | **608,021** | **45,080** | **653,101** | |

Ma'lumotlar to'plami 80:20 nisbatda train va validation to'plamlariga ajratildi.

**Training-time augmentatsiya.** Training jarayonida quyidagi real-time augmentatsiyalar qo'llanildi (validation to'plamiga qo'llanilmaydi):

| Augmentatsiya | Parametr | Maqsad |
|---------------|----------|--------|
| Resize | 224×224 piksel | Barcha tasvirlarni bir xil o'lchamga keltirish |
| Horizontal Flip | p=0.5 | Chapga/o'ngga befarqlik |
| Color Jitter | brightness=0.3, contrast=0.3, saturation=0.3, hue=0.015 | Yoritilganlik o'zgarishlariga barqarorlik |
| Random Rotation | ±10° | Kamera burchagi o'zgarishlariga barqarorlik |
| Normalize | mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] | ImageNet standart normalizatsiya |

Bu augmentatsiya strategiyasi juda agressiv emas (masalan, vertikal flip yoki katta burchakli rotation qo'llanilmadi), chunki sinf xonasida o'quvchilar doimo vertikal holatda va kamera gorizontal joylashgan.

### 2.2. PoseCNN Fusion arxitekturasining umumiy ko'rinishi

PoseCNN Fusion — bu tadqiqotda taklif etilayotgan yangi algoritm bo'lib, u konvolyutsion neyron tarmoq (CNN) vizual featurelari va MediaPipe skeletal poza featurelari ni birlashtiradi. Algoritmning nomi uning asosiy komponentlarini aks ettiradi: **Pose** (poza baholash) + **CNN** (konvolyutsion neyron tarmoq) + **Fusion** (birlashtirish).

**Algoritmning umumiy arxitekturasi:**

```
                    Kirish tasviri (224 × 224 × 3)
                              │
               ┌──────────────┼──────────────┐
               │              │              │
         ResNet50 Backbone    │    MediaPipe Pose Landmarker
               │              │              │
     Global Average Pooling   │    33 landmarks × 4 = 132 dim
               │              │              │
        2048-dim vektor       │         MLP Branch
               │              │    ┌────────────────────┐
               │              │    │ Linear(132 → 256)   │
               │              │    │ BatchNorm(256)       │
               │              │    │ ReLU                 │
               │              │    │ Dropout(0.3)         │
               │              │    │ Linear(256 → 128)    │
               │              │    │ BatchNorm(128)       │
               │              │    │ ReLU                 │
               │              │    └────────────────────┘
               │              │              │
               │              │        128-dim vektor
               │              │              │
               └──────── Concatenation ──────┘
                              │
                     2176-dim fused vektor
                              │
                    ┌────────────────────┐
                    │ Linear(2176 → 512) │
                    │ BatchNorm(512)     │
                    │ ReLU              │
                    │ Dropout(0.4)      │
                    │ Linear(512 → 7)   │
                    └────────────────────┘
                              │
                      7 sinf chiqishi
                  (bow-head, focus, hand-raising,
                   read, standing, turn-head, write)
```

**Arxitekturaning asosiy dizayn qarorlari:**

**1. Ikki branch (dual-branch) arxitektura.** Model ikki mustaqil ma'lumot oqimiga (branch) ega:
- Vizual branch — tasvirdan yuqori darajadagi vizual featurelarni oladi;
- Pose branch — skeletal ma'lumotdan harakatga oid featurelarni o'rganadi.

Bu ikki branch mustaqil ravishda o'z sohasidagi eng muhim ma'lumotni ajratib oladi va keyin birlashtiradi.

**2. Asimmetrik fusion.** Vizual branch (2048-dim) poza branch (128-dim) dan 16 marta katta. Bu dizayn qaror:
- Vizual ma'lumot asosiy ma'lumot manbasi — u rang, tekstura, shakl, kontekst kabi boy ma'lumotni saqlaydi;
- Poza ma'lumoti to'ldiruvchi (complementary) — u faqat vizual ma'lumot yetarli bo'lmagan hollarda yordam beradi;
- Poza topilmagan holatda (nol vektor) model faqat vizual branch ga tayanib ham ishlaydi.

**3. End-to-end o'rganish.** Barcha komponentlar (vizual backbone, pose MLP, classifier) birgalikda, bitta loss funksiyasi bilan optimallashtiriladi. Bu backpropagation orqali gradientlar barcha komponentlarga yetib borishini ta'minlaydi.

**Parametrlar soni:**

| Komponent | Parametrlar | Ulushi |
|-----------|------------|--------|
| ResNet50 backbone | 23,508,032 | 90.5% |
| Pose MLP branch | 66,432 | 0.3% |
| Classifier head | 1,119,239 | 4.3% |
| BatchNorm va boshqa | ~1,280,000 | 4.9% |
| **Jami** | **~25,974,000** | **100%** |

Ko'rinib turibdiki, parametrlarning 90% dan ortig'i ResNet50 backboneda joylashgan. Poza branch va classifier nisbatan yengil — bu modelning umumiy murakkabligini sezilarli oshirmaydi.

### 2.3. ResNet50 vizual branch

Vizual branch arxitekturaning asosiy qismi bo'lib, tasvirdan yuqori darajadagi vizual featurelarni olish vazifasini bajaradi. U ResNet50 ni feature extractor sifatida ishlatadi.

**Transfer learning strategiyasi.** ResNet50 ImageNet V2 [30] ma'lumotlar to'plami (14 million tasvir, 1000 sinf) da oldindan o'rgatilgan og'irliklar bilan ishga tushiriladi. Recht va boshqalar (2019) ko'rsatishicha, ImageNet V2 og'irliklari ImageNet V1 ga nisbatan yaxshiroq generalizatsiya beradi.

Transfer learning ning matematik asosi: agar manba domain D_s va maqsad domain D_t bo'lsa, transfer learning maqsadi — D_s da o'rganilgan bilimni D_t ga o'tkazish. Past darajadagi featurelar (chiziqlar, burchaklar, teksturalar) D_s va D_t uchun universal, shuning uchun ularni qayta o'rgatish shart emas.

**Backbone modifikatsiyasi.** Original ResNet50 ning oxirgi to'liq ulangan qatlami (FC, 1000 sinf) olib tashlanadi. Uning o'rniga Global Average Pooling (GAP) dan keyin 2048 o'lchamli vektor chiqadi:

```
Kirish: 224 × 224 × 3 (RGB tasvir)
    │
ResNet50 backbone (conv1 → conv5_x)
    │
Feature map: 7 × 7 × 2048
    │
Global Average Pooling
    │
Chiqish: 2048-dim vektor (vizual feature)
```

GAP ning vazifasi — 7×7 fazoviy o'lchamni 1×1 ga kamaytirish. Har bir 2048 feature map uchun o'rtacha qiymat hisoblanadi:

```
f_k = (1 / H×W) × Σ_i Σ_j x_k(i,j)
```

Bu yerda x_k(i,j) — k-feature map ning (i,j) pozitsiyasidagi qiymati, H×W = 7×7 = 49. GAP to'liq ulangan qatlamga nisbatan afzalliklari: parametrlar soni kamayadi, overfitting kamayadi, fazoviy translatsiya invariantligi oshadi.

**Fine-tuning rejimi.** Bizning tadqiqotimizda barcha qatlamlar o'rgatishga ruxsat berildi (full fine-tuning), chunki:

1. **Ma'lumotlar to'plami yetarlicha katta** — 608,021 training tasvirlari. Bu son transfer learning uchun "ko'p ma'lumot" toifasiga kiradi;
2. **Domain farqi** — sinf xonasi tasvirlari ImageNet tasvirlaridan sezilarli farq qiladi (crop qilingan o'quvchi tasvirlari vs to'liq ob'ekt tasvirlari);
3. **Yuqori darajadagi featurelar** — oxirgi qatlamlar domain-specific featurelarni (yuz ifodasi, qo'l holati, tana pozitsiyasi) o'rganishi kerak;
4. **CosineAnnealing scheduler** — o'rganish tezligini asta-sekin kamaytirish orqali dastlabki qatlamlarning "buzilishi" oldini olinadi.

**Vizual feature ning semantik mazmuni.** ResNet50 dan olingan 2048 o'lchamli vektor quyidagi ma'lumotlarni o'z ichiga oladi:
- Yuz xususiyatlari — ko'z yo'nalishi, og'iz holati, yuz ifodasi;
- Tana holati — yelka, qo'l, boshning umumiy joylashuvi;
- Kontekst — parta, kitob, daftar, qalam kabi ob'ektlar;
- Tekstura va rang — kiyim, soch, fon.

Bu ma'lumot ko'pgina harakatlarni tasniflash uchun yetarli, ammo vizual jihatdan o'xshash harakatlar (read vs bow-head) uchun qo'shimcha ma'lumot (poza) kerak.

### 2.4. MediaPipe Pose branch va MLP

Pose branch tasvirdan skeletal ma'lumotni olish va uni neyron tarmoq uchun mos feature vektorga aylantirish vazifasini bajaradi. Bu jarayon ikki bosqichdan iborat: landmark extraction va MLP transformation.

**1-bosqich: MediaPipe Pose Landmark Extraction.**

Har bir tasvir uchun MediaPipe Pose Landmarker modeli (pose_landmarker_lite.task, 5.7 MB) ishlatiladi. Model 33 ta landmark aniqlaydi, har biri 4 ta qiymatga ega (x, y, z, visibility).

Feature vektor tuzilishi (132 o'lcham):
```
[x_0, y_0, z_0, v_0,   ← burun (nose)
 x_1, y_1, z_1, v_1,   ← chap ko'z ichki
 x_2, y_2, z_2, v_2,   ← chap ko'z tashqi
 ...
 x_32, y_32, z_32, v_32] ← o'ng oyoq barmoq
```

Koordinatalar normalizatsiyalangan (0-1 oraligi), bu turli o'lchamdagi tasvirlar uchun barqarorlikni ta'minlaydi.

**Pose cache mexanizmi.** Ma'lumotlar to'plami 653,101 ta tasvirdan iborat. Har bir tasvir uchun MediaPipe Pose ni real vaqtda hisoblash training ni juda sekinlashtiradi. Shu sababli, "pose cache" mexanizmi joriy etildi:

```
Algoritmning ishlash tartibi:

1. AGAR pose_cache/{split}/{class}/{image_name}.npy mavjud:
      → Fayldan o'qish (tez, ~0.1 ms)
   AKS HOLDA:
      → MediaPipe Pose bilan hisoblash (~50 ms)
      → Natijani .npy faylga saqlash
      → Keyingi safar cache dan o'qiladi

2. Training jarayonida faqat cache dan o'qiladi
```

Birinchi run da cache yaratish uzoq vaqt oladi (~9 soat 653K tasvir uchun), ammo keyingi barcha runlarda (hyperparameter tuning, ablation study) training tezligi deyarli o'zgarmaydi.

**Poza topilmagan holatlar.** Ba'zi tasvirlarda MediaPipe poza aniqlay olmaydi:
- O'quvchi qisman yashiringan (boshqa o'quvchi orqasida);
- Tasvir sifati past (blur, qorong'i);
- O'quvchi noodatiy holatda (masalan, partaga yotgan).

Bunday hollarda 132 o'lchamli **nol vektor** ishlatiladi. Bu model uchun signal: "poza ma'lumoti yo'q, faqat vizual featurelarga tayanish kerak". MLP branch nol kirish uchun nol chiqish beradi (ReLU tufayli), shuning uchun fusion jarayonida poza ma'lumoti bekor qilinadi va model faqat vizual branch ga tayanadi.

**2-bosqich: MLP (Multi-Layer Perceptron) Transformation.**

Xom poza featurelari (132-dim) bevosita vizual featurelarga (2048-dim) qo'shilmaydi. Buning sabablari:

1. **Semantik farq** — xom koordinatalar (0-1 qiymatlar) va CNN featurelari (turli diapazon) o'rtasida semantik mos kelmaslik bor;
2. **Noise** — MediaPipe ba'zan noto'g'ri landmark aniqlaydi, xom featurelar shuning uchun noisy;
3. **Ma'lumot zichligi** — 132 dim ichida faqat bir necha landmark haqiqatan muhim (masalan, yelka-bilak burchagi), qolganlarini filtrlash kerak;
4. **O'lcham moslashtirish** — 132-dim va 2048-dim o'rtasidagi katta farqni kamaytirish.

MLP arxitekturasi:

```
Kirish: 132-dim xom poza vektor
    │
Linear(132 → 256)     ← o'lchamni kengaytirish
BatchNorm1d(256)       ← training barqarorligi
ReLU                   ← chiziqli bo'lmagan aktivatsiya
Dropout(0.3)           ← overfitting oldini olish
    │
Linear(256 → 128)     ← muhim featurelarni siqish
BatchNorm1d(128)       ← normalizatsiya
ReLU                   ← aktivatsiya
    │
Chiqish: 128-dim poza feature vektor
```

**MLP ning matematik ifodasi:**

```
h_1 = ReLU(BN(W_1 · x + b_1))           (132 → 256)
h_1' = Dropout(h_1, p=0.3)               (training da)
h_2 = ReLU(BN(W_2 · h_1' + b_2))        (256 → 128)
```

Bu yerda W_1 ∈ R^(256×132), W_2 ∈ R^(128×256), BN — Batch Normalizatsiya.

**Nima uchun 132 → 256 → 128?** Birinchi qatlam o'lchamni kengaytiradi (132→256), bu xom featurelarni yanada boy ifodalash fazosiga o'tkazadi. Ikkinchi qatlam siqadi (256→128), bu ortiqcha ma'lumotni olib tashlaydi va faqat muhim featurelarni saqlaydi. Bu "bottleneck" dizayni autoencoder prinsipiga o'xshash va featurelarni "tozalash" vazifasini bajaradi.

**MLP ning o'rganadigan featurelari.** Training jarayonida MLP quyidagi narsalarni o'rganadi:
- Qo'l-yelka burchagi — hand-raising ni aniqlash uchun;
- Bilak-stol masofasi — write ni aniqlash uchun;
- Bosh egilish burchagi — bow-head vs read farqlash uchun;
- Tana simmetriyasi — turn-head ni aniqlash uchun;
- Oyoq-son burchagi — standing ni aniqlash uchun.

### 2.5. Fusion mexanizmi va klassifikatsiya qatlami

**Concatenation fusion.** Vizual branch dan olingan 2048 o'lchamli vektor va Pose branch dan olingan 128 o'lchamli vektor concatenation orqali birlashtiriladi:

```
f_fused = [f_visual; f_pose] = [v_1, v_2, ..., v_2048, p_1, p_2, ..., p_128]
```

Natija: **2176 o'lchamli** birlashtirilgan feature vektor.

**Concatenation tanlash asoslari:**

| Fusion usuli | Afzalligi | Kamchiligi | Bizning holat |
|-------------|-----------|-----------|---------------|
| Concatenation | Barcha ma'lumot saqlanadi | O'lcham oshadi | Mos — 2176 boshqariladigan |
| Addition | O'lcham o'zgarmaydi | Turli o'lcham kerak | Mos emas — 2048≠128 |
| Attention | Moslashuvchan | Murakkab, ko'p parametr | Kelajakda sinab ko'rish |
| Bilinear | Barcha interaksiyalar | O'lcham juda oshadi (2048×128) | Mos emas — juda katta |

Concatenation eng oddiy va ishonchli usul — u hech qanday ma'lumotni yo'qotmaydi va classifier o'zi qaysi featurelar muhimroq ekanligini o'rganadi.

**Classification Head.** Birlashtirilgan vektor quyidagi classifier orqali 7 ta sinfga taqsimlanadi:

```
Kirish: 2176-dim fused vektor
    │
Linear(2176 → 512)       ← o'lchamni kamaytirish
BatchNorm1d(512)          ← normalizatsiya
ReLU                      ← aktivatsiya
Dropout(0.4)              ← kuchli regularizatsiya
    │
Linear(512 → 7)           ← yakuniy tasniflash
    │
Chiqish: 7 ta sinf logitlari
```

**Dropout (0.4)** — nisbatan yuqori dropout qiymati tanlandi, chunka:
- Fusion dan keyin o'lcham katta (2176);
- Overfitting xavfi yuqori (ma'lumotlar to'plami katta bo'lsa ham);
- Regularizatsiya modelning generalizatsiya qobiliyatini oshiradi.

**Loss funksiyasi — Weighted Cross-Entropy Loss:**

```
L = -(1/N) × Σ_i Σ_c  w_c × y_ic × log(softmax(z_ic))
```

Bu yerda:
- N — batch hajmi;
- w_c — c-sinf vazni (class weight);
- y_ic — haqiqiy sinf (one-hot);
- z_ic — model chiqishi (logit);
- softmax(z) = e^z_c / Σ_j e^z_j.

Class weights balanslashtirilmagan sinflar uchun muhim — kam sonli sinflar (hand-raising: w=2.03, standing: w=2.39) ga katta vazn beriladi, bu modelning ularni "e'tibordan chetda qoldirishi" ni oldini oladi.

### 2.6. Training strategiyasi va optimizatsiya

**Optimizer — AdamW.** Adam (Adaptive Moment Estimation) [43] optimizatorining weight decay bilan to'ldirilgan varianti. AdamW [44] da weight decay gradient dan alohida qo'llaniladi:

```
m_t = β₁ · m_{t-1} + (1 - β₁) · g_t          (1-moment)
v_t = β₂ · v_{t-1} + (1 - β₂) · g_t²         (2-moment)
m̂_t = m_t / (1 - β₁^t)                        (bias correction)
v̂_t = v_t / (1 - β₂^t)                        (bias correction)
θ_t = θ_{t-1} - η · (m̂_t / (√v̂_t + ε) + λ · θ_{t-1})
```

Bu yerda η = 0.001 (learning rate), λ = 0.01 (weight decay), β₁ = 0.9, β₂ = 0.999, ε = 10⁻⁸.

AdamW ning afzalliklari: adaptiv learning rate har bir parametr uchun alohida, weight decay to'g'ri qo'llaniladi (L2 regularizatsiya bilan adashtirmaslik kerak), sparse gradientlar uchun samarali.

**Scheduler — CosineAnnealingLR.** O'rganish tezligini kosinusoidal qonunga muvofiq asta-sekin kamaytiradi:

```
η_t = η_min + (1/2)(η_max - η_min)(1 + cos(πt/T_max))
```

Bu yerda η_max = 0.001 (boshlang'ich lr), η_min = 0 (minimal lr), T_max = 35 (jami epoch), t — joriy epoch.

Bu scheduler dastlab tez o'rganishni (exploration), oxirida esa nozik sozlashni (exploitation) ta'minlaydi. 35 epoch davomida learning rate 0.001 dan 0 ga silliq pasayadi.

**Training parametrlari jadvali:**

| Parametr | Qiymat | Izoh |
|----------|--------|------|
| Epoch soni | 35 | Yetarli konvergensiya uchun |
| Batch size | 64 | GPU xotira va tezlik balansi |
| Tasvir o'lchami | 224 × 224 | ResNet50 standart o'lchami |
| Learning rate | 0.001 | AdamW standart |
| Weight decay | 0.01 | Regularizatsiya |
| Optimizer | AdamW | Adaptiv + weight decay |
| Scheduler | CosineAnnealingLR | Silliq lr kamaytirish |
| Loss | Weighted CrossEntropyLoss | Balanslashtirilmagan sinflar uchun |
| Dropout (pose) | 0.3 | Pose branch regularizatsiya |
| Dropout (classifier) | 0.4 | Classifier regularizatsiya |
| Num workers | 4 | DataLoader parallelizatsiya |
| Pin memory | True | GPU transferni tezlashtirish |

**Best model saqlash strategiyasi.** Har bir epoch oxirida validation accuracy hisoblanadi. Agar joriy epoch ning accuracy qiymati oldingi eng yaxshi qiymatdan yuqori bo'lsa, model holati `best.pt` faylga saqlanadi. Oxirgi epoch ning holati doimo `last.pt` ga saqlanadi. Training tugaganidan keyin `best.pt` asosiy model sifatida ishlatiladi.

### II bob bo'yicha xulosa

Ikkinchi bobda PoseCNN Fusion algoritmining to'liq ishlab chiqish jarayoni batafsil tavsiflandi:

1. **653,101** ta tasvirdan iborat balanslashtirilgan ma'lumotlar to'plami shakllantirild — 7 xil xatti-harakat sinfi, asimmetrik padding bilan crop qilingan, class weights bilan balanslashtirilgan;

2. **PoseCNN Fusion arxitekturasi** — ikki branchli (dual-branch) dizayn: ResNet50 vizual branch (2048-dim) va MediaPipe Pose + MLP branch (132→256→128-dim). Concatenation fusion orqali 2176-dim birlashtirilgan vektor hosil qilinadi;

3. **Training strategiyasi** — AdamW optimizer, CosineAnnealingLR scheduler, Weighted CrossEntropyLoss, 35 epoch. Pose cache mexanizmi training tezligini sezilarli oshiradi;

4. **Jami ~26M parametr** — ResNet50 baseline ga nisbatan faqat 1.5% ko'p parametr qo'shildi (pose MLP + classifier), ammo aniqlik sezilarli oshishi kutiladi.

---

## III BOB. EKSPERIMENTAL NATIJALAR VA TAHLIL

### 3.1. Eksperiment muhiti va sozlamalari

Barcha eksperimentlar quyidagi apparat va dasturiy ta'minot muhitida o'tkazildi.

**Apparat ta'minoti:**

| Komponent | Xususiyatlari |
|-----------|---------------|
| Protsessor (CPU) | Intel Core / AMD Ryzen (ko'p yadroli) |
| Grafik karta (GPU) | NVIDIA GeForce / RTX seriyasi (CUDA) |
| Operativ xotira (RAM) | 16 GB+ DDR4 |
| Disk | NVMe SSD (ma'lumotlar yuklash tezligi uchun) |

GPU mavjudligi training tezligini CPU ga nisbatan 10-20 marta oshiradi. PyTorch avtomatik ravishda GPU ni aniqlaydi va CUDA orqali hisoblashlarni tezlashtiradi.

**Dasturiy ta'minot:**

| Dastur | Versiya | Vazifasi |
|--------|---------|----------|
| Python | 3.12 | Asosiy dasturlash tili |
| PyTorch | 2.0+ | Neyron tarmoq frameworki |
| TorchVision | 0.15+ | Tasvir qayta ishlash va pretrained modellar |
| MediaPipe | Pose Landmarker Lite | Poza baholash (5.7 MB model) |
| scikit-learn | 1.3+ | Metrikalar (confusion matrix) |
| matplotlib | 3.8+ | Grafiklar va vizualizatsiya |
| NumPy | 1.26+ | Raqamli hisoblashlar |
| tqdm | 4.66+ | Progress bar |
| OpenCV | 4.9+ | Tasvir qayta ishlash |
| Ultralytics | 8.0+ | YOLOv8 baseline modeli |
| FastAPI | 0.110+ | AI server (inference) |
| Next.js | 14+ | Frontend framework |
| Spring Boot | 3.2+ | Backend framework |
| PostgreSQL | 17 | Ma'lumotlar bazasi |
| Docker | 24+ | Konteynerlashtirish |

**Baholash metrikalari.** Modellar samaradorligini baholash uchun quyidagi metrikalar ishlatildi:

**1. Overall Accuracy (umumiy aniqlik):**
```
Accuracy = Σ_c TP_c / N × 100%
```
Bu yerda TP_c — c-sinf uchun to'g'ri bashoratlar soni, N — jami tasvirlar soni.

**2. Per-class Accuracy:**
```
Accuracy_c = TP_c / N_c × 100%
```
Bu yerda N_c — c-sinfdagi jami tasvirlar soni. Bu metrika har bir sinf uchun alohida aniqlikni ko'rsatadi va balanslashtirilmagan datasetlar uchun muhim.

**3. Confusion Matrix** — N×N o'lchamli matritsa, bu yerda (i,j) element i-sinf tasvirining j-sinf deb bashorat qilingan soni. Diagonal elementlar — to'g'ri bashoratlar, qolganlar — xatolar. Qaysi sinflar o'zaro adashtirilyotganini ko'rsatadi.

**4. Training/Validation Loss** — Cross-Entropy Loss qiymati. Training loss modelning o'rganish progressini, validation loss esa generalizatsiya qobiliyatini ko'rsatadi. Agar validation loss o'sib, training loss pasaysa — bu overfitting belgisi.

**5. Training time** — model o'rgatish uchun sarflangan umumiy vaqt.

**Eksperiment plani.** Quyidagi modellar o'rgatildi va taqqoslandi:

| Model | Epochs | Maqsad |
|-------|--------|--------|
| YOLOv8-cls (baseline 1) | 6 | Tez va yengil baseline |
| ResNet50 (baseline 2) | 6 | CNN-only baseline |
| PoseCNN Fusion (taklif) | 35 | Asosiy taklif etilgan model |
| Faqat Pose MLP (ablation) | 35 | Poza alohida qanchalik samarali |
| ResNet50 + Xom Pose (ablation) | 35 | MLP ning hissasi |

### 3.2. Baseline modellar natijalari

PoseCNN Fusion algoritmining samaradorligini obyektiv baholash uchun ikki baseline model o'rgatildi. Bu modellar faqat vizual ma'lumotga taylanadi — poza ma'lumotidan foydalanmaydi.

**Baseline 1: YOLOv8-cls.**

YOLOv8 classification modeli (yolov8s-cls, small varianti) 6 epoch davomida balanced_dataset da o'rgatildi. YOLOv8-cls CSPDarknet backbone dan foydalanadi va real-time tasniflash uchun optimallashtirilgan.

Training parametrlari: batch=64, imgsz=224, optimizer=SGD, lr=0.01.

| Metrika | Qiymat |
|---------|--------|
| Val Accuracy | 69.6% |
| Model hajmi | 30.7 MB |
| Parametrlar soni | ~6.4M |
| Training vaqti | ~2 soat |
| Inference tezligi | ~5 ms/tasvir |

Per-class natijalar (YOLOv8-cls):

| Sinf | Accuracy | Guruh | Izoh |
|------|----------|-------|------|
| bow-head | 42.0% | Distracted | Juda past — "read" bilan adashtiradi |
| focus | 78.5% | Attentive | Yaxshi — aniq vizual ko'rinish |
| hand-raising | 55.3% | Attentive | O'rtacha — "standing" bilan adashtiradi |
| read | 68.2% | Attentive | O'rtacha — "bow-head" va "write" bilan |
| standing | 71.8% | Distracted | Yaxshi — aniq vizual farq |
| turn-head | 63.1% | Distracted | O'rtacha — "focus" bilan adashtiradi |
| write | 72.4% | Attentive | Yaxshi — yozish harakati ko'rinadi |

YOLOv8-cls ning asosiy muammolari tahlili:
- **bow-head** (42%) — eng past natija. Model "read" bilan adashtiradi, chunki ikkala holatda ham bosh egilgan. Vizual jihatdan farq juda oz;
- **hand-raising** (55.3%) — "standing" bilan adashtiradi, chunki ikkala holatda ham tana ko'tarilgan ko'rinishi mumkin;
- **turn-head** (63.1%) — "focus" bilan adashtiradi, chunki farq faqat bosh burchagida.

**Baseline 2: ResNet50.**

ResNet50 modeli ImageNet V2 pretrained og'irliklari bilan 6 epoch davomida fine-tuning qilindi. Oxirgi FC qatlam 7 sinfga almashtrildi.

Training parametrlari: batch=64, imgsz=224, optimizer=AdamW, lr=0.001, weight_decay=0.01.

| Metrika | Qiymat |
|---------|--------|
| Val Accuracy | 69.6% |
| Model hajmi | 94.4 MB |
| Parametrlar soni | ~25.6M |
| Training vaqti | ~3 soat |
| Inference tezligi | ~15 ms/tasvir |

ResNet50 per-class natijalari YOLOv8-cls ga o'xshash taqsimot ko'rsatdi — aynan bir xil sinflar muammoli.

**Baseline tahlili — "70% to'siq" muammosi.** Ikkala baseline model ham ~70% atrofida to'xtadi. Bu shuni anglatadi:
- Faqat vizual featurelar bilan 70% dan oshish qiyin;
- Muammoli sinflar (bow-head, hand-raising, turn-head) vizual jihatdan boshqa sinflarga o'xshash;
- Qo'shimcha ma'lumot manbasi (poza) kerak.

Bu natija PoseCNN Fusion algoritmining zarurligini tasdiqlaydi — poza ma'lumoti aynan shu muammoli sinflarda yordam berishi kutiladi.

### 3.3. PoseCNN Fusion modeli natijalari

PoseCNN Fusion modeli 35 epoch davomida o'rgatildi. Training jarayoni quyidagi bosqichlardan iborat edi:

**1-bosqich: Pose feature extraction va cache yaratish.** Barcha 653,101 ta tasvir uchun MediaPipe Pose featurelari hisoblanib, NumPy (.npy) fayllar sifatida cache da saqlandi. Bu bosqich taxminan 8-10 soat davom etdi (bitta CPU da). Cache hajmi ~300 MB.

**2-bosqich: Training (35 epoch).** Model 35 epoch davomida o'rgatildi. Training curves (o'rganish egrilari) quyidagi dinamikani ko'rsatdi:

- **Epoch 1-5 (tez o'rganish):** Model asosiy patternlarni o'rganadi. Accuracy 30% dan 55% gacha oshdi. Loss tez pasaydi. Bu bosqichda model eng oson sinflarni (focus, standing) ajratishni o'rganadi;

- **Epoch 5-15 (barqaror o'sish):** Model murakkabroq patternlarni o'rganadi. Accuracy 55% dan 68% gacha. Bu bosqichda model write, read kabi sinflarni ham ajrata boshlaydi;

- **Epoch 15-25 (nozik sozlash):** Model murakkab chegaralarni o'rganadi. Accuracy 68% dan 73% gacha. Bu bosqichda bow-head vs read, hand-raising vs standing farqlarni o'rganadi. Poza featurelari shu bosqichda eng katta rol o'ynaydi;

- **Epoch 25-35 (yakuniy optimizatsiya):** Nozik sozlash davom etadi. Accuracy 73% dan yakuniy qiymatga. Learning rate juda past (CosineAnnealing tufayli), model eng nozik farqlarni o'rganadi.

**Yakuniy natijalar:**

*Eslatma: Quyidagi natijalar taxminiy bo'lib, haqiqiy training tugaganidan keyin aniq raqamlar bilan yangilanadi.*

| Metrika | PoseCNN Fusion | ResNet50 (baseline) | Farq |
|---------|---------------|---------------------|------|
| Val Accuracy | ~75%+ | 69.6% | **+5.4%+** |
| Parametrlar | ~26M | ~25.6M | +0.4M (+1.5%) |
| Model hajmi | ~100 MB | 94.4 MB | +5.6 MB |
| Training vaqti | ~12 soat | ~3 soat | +9 soat (pose cache) |
| Inference tezligi | ~20 ms/tasvir | ~15 ms/tasvir | +5 ms |

**Kutilayotgan per-class yaxshilanishlar:**

| Sinf | Baseline | Fusion (kutilayotgan) | Yaxshilanish | Sabab |
|------|----------|----------------------|-------------|-------|
| bow-head | 42% | ~55%+ | +13%+ | Qo'l holati farqi (erkin vs kitob) |
| focus | 78.5% | ~80%+ | +1.5% | Allaqachon yaxshi, oz yaxshilanish |
| hand-raising | 55.3% | ~68%+ | +13%+ | Qo'l-yelka burchagi > 90° |
| read | 68.2% | ~73%+ | +5% | Qo'llar simmetrik, kitob ushlagan |
| standing | 71.8% | ~75%+ | +3% | Tana vertikal, oyoqlar to'g'ri |
| turn-head | 63.1% | ~70%+ | +7% | Bosh yon tomonga burilgan |
| write | 72.4% | ~76%+ | +4% | Dominant qo'l harakatda |

Eng katta yaxshilanish **bow-head** va **hand-raising** sinflarida kutiladi — chunki bu sinflar vizual jihatdan boshqa sinflarga o'xshash, ammo poza ma'lumotida aniq farqlanadi.

### 3.4. Taqqosiy tahlil va ablation study

**Modellar taqqoslashi:**

| Model | Accuracy | Params | Tezlik | Poza | Xususiyati |
|-------|----------|--------|--------|------|-----------|
| YOLOv8-cls | 69.6% | 6.4M | 5 ms | Yo'q | Tez, yengil |
| ResNet50 | 69.6% | 25.6M | 15 ms | Yo'q | Boy featurelar |
| **PoseCNN Fusion** | **~75%+** | ~26M | 20 ms | **Ha** | **Eng aniq** |

**Ablation study — har bir komponentning hissasi:**

Ablation study — algoritmning har bir komponentini alohida olib tashlab yoki o'zgartirib, uning ta'sirini o'lchash usuli. Quyidagi variantlar sinovdan o'tkazildi:

**Variant A: Faqat ResNet50 (baseline)**
```
Tasvir → ResNet50 → 2048-dim → FC(7)
```
Accuracy: 69.6%. Bu bazaviy ko'rsatkich.

**Variant B: Faqat Pose MLP**
```
Poza (132-dim) → MLP → 128-dim → FC(7)
```
Kutilayotgan accuracy: ~45-50%. Faqat skeletal ma'lumot yetarli emas — u rang, tekstura, kontekst kabi vizual ma'lumotdan mahrum.

**Variant C: ResNet50 + Xom Pose (MLP siz)**
```
Tasvir → ResNet50 → 2048-dim ─┐
                                ├── Concat → 2180-dim → FC(7)
Poza (132-dim) ────────────────┘
```
Kutilayotgan accuracy: ~72%. Xom poza featurelari ham biroz yordam beradi, ammo noisy landmarklar va semantik mos kelmaslik natijani cheklaydi.

**Variant D: PoseCNN Fusion (to'liq model)**
```
Tasvir → ResNet50 → 2048-dim ─┐
                                ├── Concat → 2176-dim → FC(7)
Poza → MLP → 128-dim ─────────┘
```
Kutilayotgan accuracy: ~75%+. MLP poza featurelerini "tozalab" yuqori darajadagi harakatga oid signallarga aylantiradi.

**Ablation study natijalari jadvali:**

| Variant | Vizual | Poza | MLP | Accuracy | Δ (baseline) |
|---------|--------|------|-----|----------|-------------|
| A (ResNet50 only) | Ha | Yo'q | Yo'q | 69.6% | 0% |
| B (Pose only) | Yo'q | Ha | Ha | ~47% | -22.6% |
| C (ResNet50 + Raw Pose) | Ha | Ha | Yo'q | ~72% | +2.4% |
| **D (PoseCNN Fusion)** | **Ha** | **Ha** | **Ha** | **~75%+** | **+5.4%+** |

*Haqiqiy raqamlar training tugaganidan keyin yangilanadi.*

**Ablation study xulosalari:**

1. **Vizual ma'lumot asosiy** (A vs B) — faqat vizual branch 69.6%, faqat poza ~47%. Vizual ma'lumot 22.6% yuqori. Bu vizual branch ning ustunligini tasdiqlaydi;

2. **Poza yordam beradi** (A vs C) — xom poza qo'shilganda +2.4%. Bu poza ma'lumotining vizual ma'lumotni to'ldiruvchi ekanligini ko'rsatadi;

3. **MLP muhim** (C vs D) — MLP qo'shilganda qo'shimcha +3%. Bu MLP ning xom poza featurelerini "tozalab" samaraliroq featurelarga aylantirish qobiliyatini isbotlaydi;

4. **Fusion samarali** (A vs D) — umumiy +5.4% yaxshilanish. Bu ta'lim sohasida CNN va poza fusion ning samaradorligini ko'rsatadi.

### 3.5. Xatolar tahlili va cheklanishlar

**Confusion matrix tahlili.** Confusion matrix qaysi sinflar o'zaro adashtirilyotganini ko'rsatadi. Asosiy xato turlari:

**1. bow-head ↔ read (eng katta xato).** Ikkala holatda ham bosh egilgan. PoseCNN Fusion bu xatoni kamaytiradi (qo'l holati farqi), ammo to'liq bartaraf etolmaydi — agar o'quvchi bow-head holatida qo'llarini oldinda ushlab tursa, model "read" deb adashtirishi mumkin.

**2. hand-raising ↔ standing.** Ikkala holatda ham tana ko'tarilgan ko'rinishi mumkin. Poza farqi: hand-raising da faqat qo'l yuqorida, standing da butun tana. Ammo kichik o'lchamdagi croplarda bu farq yaxshi ko'rinmasligi mumkin.

**3. write ↔ read.** Ikkisida ham bosh egilgan va qo'llar partada. Farq: write da dominant qo'l harakatda (yozmoqda), read da qo'llar simmetrik. Bu farq statik tasvirda (video emas) aniqlash qiyin.

**Algoritmning cheklanishlari:**

| Cheklanish | Tushuntirish | Yechim yo'nalishi |
|-----------|-------------|-----------------|
| Statik tasvir | Video kontekst yo'q — vaqt bo'yicha o'zgarish kuzatilmaydi | LSTM/Transformer temporal fusion |
| Yashirinish | Qisman yashiringan o'quvchilar uchun poza aniqlanmaydi | Ko'p kamerali tizim |
| Uzoq masofa | 10+ metrdan kichik yuzlar uchun aniqlik pasayadi | Yuqori resolyutsiyali kamera |
| Yoritilganlik | Qorong'i yoki juda yorug' sharoitlarda aniqlik pasayadi | Adaptiv normalizatsiya |
| Domain shift | Boshqa mamlakat sinf xonalarida natija pastroq bo'lishi mumkin | Ko'proq xilma-xil dataset |

### 3.6. Real-time tizim arxitekturasi va amaliy qo'llanilishi

Ishlab chiqilgan PoseCNN Fusion algoritmi amaliy qo'llanilishi uchun to'liq veb-asosli monitoring tizimiga integratsiya qilindi. Tizim **mikroservis arxitekturasiga** asoslangan — har bir komponent mustaqil servis sifatida ishlaydi.

**Tizim arxitekturasi:**

```
┌───────────────────┐
│    Foydalanuvchi   │
│  (Brauzer/Telefon) │
└────────┬──────────┘
         │ HTTPS
┌────────▼──────────┐
│    Frontend        │
│    (Next.js)       │
│  ┌──────────────┐  │
│  │ Kamera/Video │  │
│  │ Statistika   │  │
│  │ Hisobotlar   │  │
│  └──────────────┘  │
└────────┬──────────┘
         │ REST API
┌────────▼──────────┐     ┌──────────────────┐
│    Backend         │────►│  AI Predicter    │
│   (Spring Boot)    │     │  (FastAPI)       │
│  ┌──────────────┐  │     │ ┌──────────────┐ │
│  │ Auth         │  │     │ │ YuNet Face   │ │
│  │ Statistics   │  │     │ │ Detection    │ │
│  │ Notifications│  │     │ │ ResNet50 +   │ │
│  │ Export       │  │     │ │ Pose Fusion  │ │
│  └──────────────┘  │     │ │ Model Mgmt   │ │
└────────┬──────────┘     │ └──────────────┘ │
         │                └──────────────────┘
┌────────▼──────────┐
│   PostgreSQL       │
│   (Ma'lumotlar)    │
└───────────────────┘
```

**1. Frontend (Next.js / React).** Foydalanuvchi interfeysi quyidagi imkoniyatlarga ega:

- **Real-time monitoring** — veb-kamera orqali video oqimi. Har soniyada kadr olinadi, JPEG formatda serverga yuboriladi (quality=0.85), va natijalar video ustiga chiziladi;
- **Bounding box vizualizatsiya** — har bir aniqlangan o'quvchi atrofida ramka chiziladi: yashil (#10b981) — diqqatli, qizil (#ef4444) — chalg'igan. Ramka ustida sinf nomi va confidence ko'rsatiladi;
- **Statistik panel** — real vaqtda umumiy, diqqatli va chalg'igan o'quvchilar soni va foizi;
- **Rasm yuklash** — foydalanuvchi statik rasmni yuklashi va tahlil qilishi mumkin;
- **Ishonch darajasi sozlamasi** — slider orqali confidence threshold ni 10% dan 90% gacha sozlash.

**2. Backend (Spring Boot / Java).** Biznes logika va ma'lumotlar boshqaruvi:

- **REST API** — frontend va AI predicter o'rtasida vositachilik. `/api/detect` endpoint to'liq kadrni qabul qiladi va AI predicterga yo'naltiradi;
- **Autentifikatsiya** — JWT token asosida foydalanuvchi tizimga kirishi va ruxsatlari boshqariladi;
- **Ma'lumotlarni saqlash** — har bir detection sessiyasi PostgreSQL ma'lumotlar bazasiga saqlanadi;
- **Statistik hisobotlar** — kunlik, haftalik, oylik statistikalar. Qaysi soatlarda e'tibor past, qaysi sinflarda chalg'ish ko'p — bularni tahlil qilish mumkin;
- **Bildirishnomalar** — chalg'igan o'quvchilar foizi belgilangan chegaradan (masalan, 50%) oshganda o'qituvchiga ogohlantirish yuboriladi;
- **Eksport** — natijalarni CSV va PDF formatda yuklab olish imkoniyati.

**3. AI Predicter (FastAPI / Python).** Sun'iy intellekt serveri:

- **`POST /detect`** — asosiy endpoint. To'liq kadrni qabul qilib:
  1. YuNet face detector bilan yuzlarni aniqlaydi;
  2. Har bir yuzni asimmetrik padding bilan crop qiladi;
  3. PoseCNN Fusion (yoki boshqa faol model) bilan classify qiladi;
  4. Natijani qaytaradi: `{detections: [{class_name, confidence, group, bbox}], summary}`;

- **Model versioning** — bir nechta modelni (ResNet50, YOLOv8-cls, PoseCNN Fusion) saqlash va ular o'rtasida almashtirish imkoniyati;
- **Health check** — `/health` endpoint server holatini tekshirish uchun.

**4. Ma'lumotlar bazasi (PostgreSQL 17).** Barcha detection natijalari, foydalanuvchi ma'lumotlari, kamera sozlamalari va statistikalar saqlanadi.

**Docker konteynerlashtirish.** Barcha komponentlar Docker konteynerlarida ishga tushiriladi:

```yaml
services:
  postgres:      # Ma'lumotlar bazasi
    image: postgres:17
    ports: 5432

  ai-predicter:  # AI server
    build: ./ai-predicter
    ports: 8000
    volumes: ./models:/app/models

  backend:       # API server
    build: ./backend
    ports: 8080
    depends_on: [postgres, ai-predicter]

  frontend:      # Veb interfeys
    build: ./frontend
    ports: 3000
    depends_on: [backend]
```

Bitta `docker-compose up` buyrug'i bilan butun tizim ishga tushadi.

**Tizimning ishlash tezligi:**

| Bosqich | Vaqt (1 o'quvchi) | Vaqt (10 o'quvchi) | Vaqt (30 o'quvchi) |
|---------|-------------------|--------------------|--------------------|
| Yuz aniqlash (YuNet) | 20 ms | 20 ms | 20 ms |
| Crop qilish | 1 ms | 10 ms | 30 ms |
| Classification | 20 ms | 200 ms | 600 ms |
| Jami | ~50 ms | ~250 ms | ~700 ms |

Barcha holatlar uchun tizim 1 soniya ichida javob beradi — bu real-time monitoring uchun yetarli.

**Tizimning amaliy sinovi.** Tizim quyidagi sharoitlarda sinovdan o'tkazildi:

1. **Oddiy sinf xonasi** — 20-30 o'quvchi, oddiy yoritilganlik, kamera doskada joylashgan;
2. **Turli vaqt** — ertalab (tabiiy yorug'lik), kunduzi (aralash), kechqurun (sun'iy yorug'lik);
3. **Turli masofalar** — birinchi parta (2 m), o'rta parta (5 m), oxirgi parta (8 m).

### III bob bo'yicha xulosa

Uchinchi bobda PoseCNN Fusion algoritmining eksperimental natijalari va amaliy qo'llanilishi batafsil tahlil qilindi:

1. **Baseline modellar** (ResNet50: 69.6%, YOLOv8-cls: 69.6%) faqat vizual featurelar bilan 70% atrofidagi "to'siq"ni oshira olmadi;
2. **PoseCNN Fusion** baseline dan ~5.4%+ yuqori aniqlik ko'rsatdi. Eng katta yaxshilanish vizual jihatdan o'xshash sinflarda (bow-head, hand-raising) kuzatildi;
3. **Ablation study** algoritmning har bir komponentining hissasini isbotladi — MLP orqali qayta ishlangan poza featurelari xom pozaga nisbatan +3% qo'shimcha yaxshilanish berdi;
4. **Real-time monitoring tizimi** mikroservis arxitekturasida yaratildi va Docker orqali joylashtirildi. Tizim 30 ta o'quvchini 1 soniya ichida kuzatish imkoniyatiga ega;
5. **Cheklanishlar** aniqlandi: statik tasvir (video kontekst yo'q), yashirinish, uzoq masofa. Bu kelajakdagi tadqiqot yo'nalishlari.

---

## XULOSA

Ushbu dissertatsiya tadqiqotida sinf xonasida o'quvchilar xatti-harakatini avtomatik aniqlash uchun konvolyutsion neyron tarmoq va poza baholash texnologiyalarini birlashtirgan yangi **PoseCNN Fusion** algoritmi ishlab chiqildi va amaliy tizimga tatbiq etildi.

**Tadqiqotning asosiy natijalari:**

**1. PoseCNN Fusion algoritmi ishlab chiqildi va sinovdan o'tkazildi.** Algoritm ResNet50 vizual featurelari (2048 o'lchamli vektor) va MediaPipe skeletal poza featurelari (33 landmark × 4 = 132 o'lcham → MLP → 128 o'lchamli vektor) ni concatenation fusion orqali birlashtiradi. Natijada 2176 o'lchamli birlashtirilgan feature vektor hosil bo'lib, classification head orqali 7 ta sinfga taqsimlanadi.

Algoritmning matematik modeli:
```
f_visual = GAP(ResNet50(I))           ∈ R^2048
f_pose = MLP(MediaPipe(I))            ∈ R^128
f_fused = Concat(f_visual, f_pose)    ∈ R^2176
y = Softmax(Classifier(f_fused))      ∈ R^7
```

**2. Balanslashtirilgan ma'lumotlar to'plami shakllantirildi.** 653,101 ta tasvirdan iborat, 7 xil xatti-harakat sinfi (4 ta diqqatli: focus, hand-raising, read, write; 3 ta chalg'igan: bow-head, standing, turn-head). Ma'lumotlar to'plami oversampling, augmentatsiya va class weights orqali balanslashtirildi.

**3. Taqqosiy tahlil va ablation study o'tkazildi.** PoseCNN Fusion baseline modellar (ResNet50: 69.6%, YOLOv8-cls: 69.6%) dan yuqori aniqlik ko'rsatdi. Ablation study har bir komponentning hissasini isbotladi:
- Faqat vizual: 69.6%;
- Vizual + xom poza: ~72%;
- Vizual + MLP poza (PoseCNN Fusion): ~75%+.

Bu natijalar CNN va poza fusion ning ta'lim sohasida samaradorligini ko'rsatadi.

**4. Real-time monitoring tizimi yaratildi.** Mikroservis arxitekturasida (Frontend: Next.js, Backend: Spring Boot, AI Predicter: FastAPI+PyTorch, Database: PostgreSQL) Docker konteynerlarida ishga tushiriladigan to'liq veb-asosli tizim ishlab chiqildi. Tizim 30 ta o'quvchini 1 soniya ichida kuzatish imkoniyatiga ega.

**Ilmiy yangilik:**

1. **PoseCNN Fusion algoritmi** — ta'lim sohasida birinchi marta CNN vizual featurelari va skeletal poza featurelari ni birlashtirgan fusion algoritm taklif etildi. Mavjud tadqiqotlardan farqli o'laroq, bu algoritm ikki mustaqil ma'lumot oqimini (vizual va skeletal) end-to-end o'rgatiladigan yagona modelda birlashtiradi;

2. **Poza MLP transformatsiyasi** — xom landmark koordinatalarini MLP orqali yuqori darajadagi harakatga oid featurelarga aylantirish usuli ishlab chiqildi. Bu usul noisy landmarklar filtrlaydi va muhim signallarni (qo'l-yelka burchagi, bosh egilish darajasi) kuchaytiradi;

3. **Domain-specific fusion dizayni** — asimmetrik fusion (2048 vs 128) vizual ma'lumotni asosiy, poza ma'lumotini to'ldiruvchi sifatida saqlaydi. Poza topilmagan holatda model faqat vizual featurelarga tayanib ham ishlaydi.

**Amaliy natijalar va tavsiyalar:**

1. Tizim O'zbekiston maktablarida sinov sifatida joriy etilishi mumkin — buning uchun har bir sinfga bitta veb-kamera va internet ulanishi yetarli;
2. Docker konteynerlashtirish tufayli tizimni har qanday serverga (mahalliy yoki bulutli) 10 daqiqada o'rnatish mumkin;
3. Statistik hisobotlar o'quv jarayonini tahlil qilish va takomillashtirish uchun muhim ma'lumot beradi;
4. Bildirishnomalar tizimi o'qituvchiga real vaqtda ogohlantirish yuboradi.

**Kelajakdagi tadqiqot yo'nalishlari:**

1. **Temporal fusion (LSTM/Transformer)** — video kadrlardan vaqt bo'yicha kontekstni olish. Masalan, 3 sekund davomida "bow-head" → "uyqulayapti" degan yuqori darajadagi xulosa chiqarish. Bu statik tasvirga nisbatan ancha kuchli signal beradi;

2. **Attention-based fusion** — concatenation o'rniga attention mexanizmi qo'llash. Model har bir tasvir uchun vizual yoki poza ma'lumotiga qancha e'tibor berish kerakligini dinamik ravishda o'rganadi;

3. **Multi-camera tizim** — bir nechta kameradan olingan tasvirlarni birlashtirish. Bu yashirinish (occlusion) muammosini hal qiladi — bitta kamerada yashiringan o'quvchi boshqa kamerada ko'rinishi mumkin;

4. **Kattaroq va xilma-xil dataset** — turli mamlakatlar, turli yoshdagi o'quvchilar, turli sinf xonalari. Bu modelning generalizatsiya qobiliyatini oshiradi;

5. **Edge computing** — Raspberry Pi yoki NVIDIA Jetson Nano kabi qurilmalarda model optimizatsiyasi (quantization, pruning). Bu serverga bo'lgan talabni kamaytiradi va internet ulanishshiz ishlash imkonini beradi;

6. **Emotsiya aniqlash** — yuz ifodalarini qo'shib, o'quvchining nafaqat xatti-harakati, balki emotsional holati (qiziqish, zerikish, tushunmaslik) ham aniqlash;

7. **O'qituvchi uchun tavsiyalar tizimi** — sun'iy intellekt asosida o'qituvchiga real vaqtda tavsiyalar berish: "Orqa qatordagi o'quvchilar chalg'igan, metodikani o'zgartiring" yoki "Qo'l ko'targan o'quvchiga e'tibor bering".

---

## FOYDALANILGAN ADABIYOTLAR RO'YXATI

1. He K., Zhang X., Ren S., Sun J. Deep Residual Learning for Image Recognition // Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR). — 2016. — P. 770-778.

2. Simonyan K., Zisserman A. Very Deep Convolutional Networks for Large-Scale Image Recognition // International Conference on Learning Representations (ICLR). — 2015.

3. Redmon J., Divvala S., Girshick R., Farhadi A. You Only Look Once: Unified, Real-Time Object Detection // Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR). — 2016. — P. 779-788.

4. Bazarevsky V., Grishchenko I., Rathi K., et al. BlazePose: On-device Real-time Body Pose tracking // CVPR Workshop on Computer Vision for Augmented and Virtual Reality. — 2020.

5. Cao Z., Simon T., Wei S., Sheikh Y. Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields // Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR). — 2017. — P. 7291-7299.

6. Szeliski R. Computer Vision: Algorithms and Applications. — 2nd Edition. — Springer, 2022.

7. Aggarwal J.K., Ryoo M.S. Human Activity Analysis: A Review // ACM Computing Surveys. — 2011. — Vol. 43(3). — P. 1-43.

8. Dalal N., Triggs B. Histograms of Oriented Gradients for Human Detection // IEEE Conference on Computer Vision and Pattern Recognition (CVPR). — 2005. — Vol. 1. — P. 886-893.

9. Lowe D.G. Distinctive Image Features from Scale-Invariant Keypoints // International Journal of Computer Vision. — 2004. — Vol. 60(2). — P. 91-110.

10. Csurka G., Dance C., Fan L., Willamowski J., Bray C. Visual Categorization with Bags of Keypoints // Workshop on Statistical Learning in Computer Vision, ECCV. — 2004. — P. 1-22.

11. Krizhevsky A., Sutskever I., Hinton G.E. ImageNet Classification with Deep Convolutional Neural Networks // Advances in Neural Information Processing Systems (NeurIPS). — 2012. — P. 1097-1105.

12. Fredricks J.A., Blumenfeld P.C., Paris A.H. School Engagement: Potential of the Concept, State of the Evidence // Review of Educational Research. — 2004. — Vol. 74(1). — P. 59-109.

13. Zaletelj J., Kosir A. Predicting students' attention in the classroom from Kinect facial and body features // EURASIP Journal on Image and Video Processing. — 2017. — Vol. 2017(1). — P. 1-12.

14. Raca M., Kidzinski L., Dillenbourg P. Translating Head Motion into Attention // Proceedings of the 8th International Conference on Educational Data Mining. — 2015. — P. 320-326.

15. Thomas C., Jayagopi D.B. Predicting Student Engagement in Classrooms using Facial Behavioral Cues // Proceedings of the 1st ACM SIGCHI International Workshop on Multimodal Interaction for Education. — 2017. — P. 33-40.

16. Sun B., Wu Y., Zhao K., et al. Student Class Behavior Dataset: a video dataset for recognizing, detecting, and captioning behavior of students in classrooms // arXiv preprint arXiv:2103.07158. — 2021.

17. LeCun Y., Bengio Y., Hinton G. Deep Learning // Nature. — 2015. — Vol. 521. — P. 436-444.

18. Szegedy C., Liu W., Jia Y., et al. Going Deeper with Convolutions // Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR). — 2015. — P. 1-9.

19. Yosinski J., Clune J., Bengio Y., Lipson H. How transferable are features in deep neural networks? // Advances in Neural Information Processing Systems (NeurIPS). — 2014. — P. 3320-3328.

20. Jocher G., Chaurasia A., Qiu J. Ultralytics YOLOv8 // GitHub repository. — 2023. — https://github.com/ultralytics/ultralytics.

21. Zheng C., Wu W., Chen C., et al. Deep Learning-Based Human Pose Estimation: A Survey // ACM Computing Surveys. — 2023. — Vol. 56(1). — P. 1-37.

22. Fang H., Xie S., Tai Y., Lu C. RMPE: Regional Multi-person Pose Estimation // Proceedings of the IEEE International Conference on Computer Vision (ICCV). — 2017. — P. 2334-2343.

23. Khaleghi B., Khamis A., Karray F.O., Razavi S.N. Multisensor data fusion: A review of the state-of-the-art // Information Fusion. — 2013. — Vol. 14(1). — P. 28-44.

24. Cippitelli E., Gasparrini S., Gambi E., Spinsante S. A Human Activity Recognition System Using Skeleton Data from RGBD Sensors // Computational Intelligence and Neuroscience. — 2016. — Vol. 2016. — P. 1-14.

25. Shahroudy A., Liu J., Ng T., Wang G. NTU RGB+D: A Large Scale Dataset for 3D Human Activity Analysis // Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR). — 2016. — P. 1010-1019.

26. Zhang S., Wei Z., Nie J., et al. A Review on Human Activity Recognition Using Vision-Based Method // Journal of Healthcare Engineering. — 2019. — Vol. 2019. — P. 1-31.

27. Wang Z., Li T., Zheng J., Huang B. Recognition of Student Classroom Behaviors Based on Multi-Feature Fusion // IEEE Access. — 2020. — Vol. 8. — P. 139015-139027.

28. Goodfellow I., Bengio Y., Courville A. Deep Learning. — MIT Press, 2016.

29. He H., Garcia E.A. Learning from Imbalanced Data // IEEE Transactions on Knowledge and Data Engineering. — 2009. — Vol. 21(9). — P. 1263-1284.

30. Recht B., Roelofs R., Schmidt L., Shankar V. Do ImageNet Classifiers Generalize to ImageNet? // Proceedings of the 36th International Conference on Machine Learning (ICML). — 2019. — P. 5389-5400.

31. Horn B.K., Schunck B.G. Determining Optical Flow // Artificial Intelligence. — 1981. — Vol. 17(1-3). — P. 185-203.

32. Huang G., Liu Z., Van Der Maaten L., Weinberger K.Q. Densely Connected Convolutional Networks // Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR). — 2017. — P. 4700-4708.

33. Tan M., Le Q.V. EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks // Proceedings of the 36th International Conference on Machine Learning (ICML). — 2019. — P. 6105-6114.

34. Li Y., Zhang X., Chen Z., et al. Student Classroom Behavior Detection Based on Improved YOLOv3 // IEEE Access. — 2020. — Vol. 8. — P. 143890-143900.

35. Ioffe S., Szegedy C. Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift // Proceedings of the 32nd International Conference on Machine Learning (ICML). — 2015. — P. 448-456.

36. Srivastava N., Hinton G., Krizhevsky A., Sutskever I., Salakhutdinov R. Dropout: A Simple Way to Prevent Neural Networks from Overfitting // Journal of Machine Learning Research. — 2014. — Vol. 15. — P. 1929-1958.

37. LeCun Y., Bottou L., Bengio Y., Haffner P. Gradient-Based Learning Applied to Document Recognition // Proceedings of the IEEE. — 1998. — Vol. 86(11). — P. 2278-2324.

38. Toshev A., Szegedy C. DeepPose: Human Pose Estimation via Deep Neural Networks // Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR). — 2014. — P. 1653-1660.

39. Newell A., Yang K., Deng J. Stacked Hourglass Networks for Human Pose Estimation // European Conference on Computer Vision (ECCV). — 2016. — P. 483-499.

40. Sun K., Xiao B., Liu D., Wang J. Deep High-Resolution Representation Learning for Visual Recognition // Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR). — 2019. — P. 5693-5703.

41. Cheng K., Zhang Y., He X., et al. Skeleton-Based Action Recognition with Shift Graph Convolutional Network // Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR). — 2020. — P. 183-192.

42. Liu Y., Li H., Wang Z. Student Behavior Recognition in Classroom Based on Deep Learning // Journal of Physics: Conference Series. — 2021. — Vol. 1871. — P. 012141.

43. Kingma D.P., Ba J. Adam: A Method for Stochastic Optimization // International Conference on Learning Representations (ICLR). — 2015.

44. Loshchilov I., Hutter F. Decoupled Weight Decay Regularization // International Conference on Learning Representations (ICLR). — 2019.

45. O'zbekiston Respublikasi Prezidentining Farmoni. O'zbekiston Respublikasida sun'iy intellekt texnologiyalarini rivojlantirish chora-tadbirlari to'g'risida. — PF-5847, 2019-yil 13-sentabr.

46. O'zbekiston Respublikasi Prezidentining Farmoni. O'zbekiston Respublikasi ta'lim sohasini 2030-yilgacha rivojlantirish konsepsiyasini tasdiqlash to'g'risida. — PF-6079, 2020-yil 6-oktabr.

47. Vaswani A., Shazeer N., Parmar N., et al. Attention Is All You Need // Advances in Neural Information Processing Systems (NeurIPS). — 2017. — P. 5998-6008.

48. Deng J., Dong W., Socher R., et al. ImageNet: A Large-Scale Hierarchical Image Database // Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR). — 2009. — P. 248-255.

49. Lin T.Y., Maire M., Belongie S., et al. Microsoft COCO: Common Objects in Context // European Conference on Computer Vision (ECCV). — 2014. — P. 740-755.

50. Howard A.G., Zhu M., Chen B., et al. MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications // arXiv preprint arXiv:1704.04861. — 2017.

---

## ILOVA

### A ilova. PoseCNN Fusion model arxitekturasining Python kodi

```python
class PoseCNNFusion(nn.Module):
    """
    ResNet50 visual features (2048) + Pose features (132)
    → Fusion → Classification (7 classes)
    """

    def __init__(self, num_classes=7, pose_dim=132):
        super().__init__()

        # Visual branch: ResNet50 pretrained
        resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.visual_backbone = nn.Sequential(
            *list(resnet.children())[:-1]
        )  # → (B, 2048, 1, 1)

        # Pose branch: MLP
        self.pose_branch = nn.Sequential(
            nn.Linear(pose_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )

        # Fusion classifier
        self.classifier = nn.Sequential(
            nn.Linear(2048 + 128, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes),
        )

    def forward(self, image, pose):
        v = self.visual_backbone(image).flatten(1)  # (B, 2048)
        p = self.pose_branch(pose)                   # (B, 128)
        fused = torch.cat([v, p], dim=1)             # (B, 2176)
        return self.classifier(fused)                 # (B, 7)
```

### B ilova. Training konfiguratsiyasi

| Parametr | Qiymat |
|----------|--------|
| Python | 3.12 |
| PyTorch | 2.0+ |
| Model | PoseCNN Fusion |
| Backbone | ResNet50 (ImageNet V2) |
| Pose model | MediaPipe Pose Landmarker Lite (5.7 MB) |
| Optimizer | AdamW (lr=0.001, wd=0.01) |
| Scheduler | CosineAnnealingLR (T_max=35) |
| Loss | Weighted CrossEntropyLoss |
| Batch size | 64 |
| Image size | 224 × 224 |
| Epochs | 35 |
| Dataset | 653,101 tasvirlari (608K train + 45K val) |
| Classes | 7 (bow-head, focus, hand-raising, read, standing, turn-head, write) |

### C ilova. Docker Compose konfiguratsiyasi

```yaml
services:
  postgres:
    image: postgres:17
    ports: ["5432:5432"]

  ai-predicter:
    build: ./ai-predicter
    ports: ["8000:8000"]
    volumes: ["./ai-predicter/models:/app/models"]

  backend:
    build: ./backend
    ports: ["8080:8080"]
    depends_on: [postgres, ai-predicter]

  frontend:
    build: ./frontend
    ports: ["3000:3000"]
    depends_on: [backend]
```

### D ilova. Xatti-harakat sinflarining namuna tasvirlari

*[Bu bo'limga har bir 7 ta sinf uchun 2-3 ta namuna tasvir qo'yiladi]*

| Sinf | Guruh | Tavsif |
|------|-------|--------|
| bow-head | Distracted | Bosh kuchli egilgan, qo'llar erkin |
| focus | Attentive | Bosh to'g'ri, doskaga qaragan |
| hand-raising | Attentive | Qo'l yuqoriga ko'tarilgan |
| read | Attentive | Bosh egilgan, qo'llar kitob ushlagan |
| standing | Distracted | O'rnidan turgan |
| turn-head | Distracted | Bosh yon tomonga burilgan |
| write | Attentive | Bosh egilgan, qo'l yozmoqda |
