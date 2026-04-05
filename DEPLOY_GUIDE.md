# دليل رفع المنصة الأكاديمية الذكية

## أولاً: رفع المشروع إلى GitHub

### 1. إنشاء مستودع جديد على GitHub
- افتح [github.com/new](https://github.com/new)
- اكتب اسم المستودع (مثال: `Academic_platform`)
- اختر **Public**
- لا تضع علامة على Add a README
- لا تضع علامة على Add .gitignore
- انقر **Create repository**

### 2. تجهيز المشروع محلياً
افتح Terminal أو CMD في مجلد المشروع ونفّذ:

```bash
cd E:\Strudin\ali_J
```

### 3. تهيئة Git
```bash
git init
git config user.name "اسم المستخدم على GitHub"
git config user.email "بريدك@example.com"
```

### 4. إضافة الملفات
```bash
git add app.py requirements.txt tahoma.ttf "فهرس الرسائل الجامعية.xlsx" "واجهة النتائج المحتملة.html"
```

### 5. عمل Commit
```bash
git commit -m "المنصة الأكاديمية الذكية"
```

### 6. ربط المستودع ورفع الملفات
استبدل `USERNAME` و `REPO_NAME` باسم حسابك واسم المستودع:

```bash
git branch -M main
git remote add origin https://github.com/USERNAME/REPO_NAME.git
git push -u origin main
```

### 7. عند التحديث مستقبلاً
بعد أي تعديل على الملفات:

```bash
git add -A
git commit -m "وصف التعديل"
git push
```

---

## ثانياً: نشر المنصة على Streamlit Cloud (مجاني)

### 1. إنشاء حساب
- اذهب إلى [share.streamlit.io](https://share.streamlit.io)
- انقر **Continue with GitHub**
- سجّل الدخول بحساب GitHub الخاص بك
- اسمح بالوصول إلى المستودعات

### 2. إنشاء تطبيق جديد
- انقر **Create app** في الزاوية العلوية
- اختر **Yup, I have an app**

### 3. ملء البيانات
| الحقل | القيمة |
|-------|--------|
| Repository | `USERNAME/REPO_NAME` |
| Branch | `main` |
| Main file path | `app.py` |

### 4. النشر
- انقر **Deploy**
- انتظر 2-5 دقائق حتى يكتمل البناء
- ستحصل على رابط مثل: `https://username-reponame.streamlit.app`

### 5. إعادة التشغيل عند التحديث
- Streamlit Cloud يتحدث تلقائياً عند عمل push على GitHub
- أو يمكنك إعادة التشغيل يدوياً:
  - افتح التطبيق على streamlit.app
  - انقر **Manage app** في الزاوية السفلية اليمنى
  - انقر **Reboot app**

---

## ملاحظات مهمة

### ملفات المشروع المطلوبة
| الملف | الوصف |
|-------|-------|
| `app.py` | التطبيق الرئيسي |
| `requirements.txt` | المكتبات المطلوبة |
| `tahoma.ttf` | خط عربي لسحابة الكلمات |
| `فهرس الرسائل الجامعية.xlsx` | قاعدة البيانات |

### بيانات تسجيل دخول المشرف
| الحقل | القيمة |
|-------|--------|
| اسم المستخدم | `admin` |
| كلمة المرور | `admin2024` |

لتغيير بيانات الدخول: عدّل المتغيرات `ADMIN_USER` و `ADMIN_PASS` في ملف `app.py`

### مفتاح Groq API
- احصل على مفتاح مجاني من [console.groq.com/keys](https://console.groq.com/keys)
- أدخله في الشريط الجانبي للمنصة بعد التشغيل
