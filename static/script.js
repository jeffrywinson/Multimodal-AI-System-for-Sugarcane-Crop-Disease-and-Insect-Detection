// static/script.js

// ============== TRANSLATION SETUP ==============
let currentLanguage = 'en';

// --- Comprehensive translation object using CORRECTED short keys ---
const translations = {
    en: {
        // Static UI Text
        headerTitle: "🧑‍🌾 Agri-AI Diagnostics",
        headerLead: "Upload a crop or insect image for a complete analysis.",
        translateButton: "Translate",
        step1Title: "Step 1: Upload Image",
        uploadLabel: "🌿 Upload Image (Crop or Insect)",
        analyzeButton: "Analyze Image",
        loadingSpinner: "Loading...",
        step2Title: "Step 2: Answer a Few Questions",
        step2Lead: "Please provide additional information based on the image below and your observations.",
        getDiagnosisButton: "Get Final Diagnosis",
        resultsTitle: "Final Analysis",
        resetButton: "Analyze Another Crop",
        diseaseSectionTitle: "Dead-Heart Symptoms",
        insectSectionTitle: "Larva Symptoms",
        yes: "Yes",
        no: "No",
        diseaseAnalysisTitle: "🌿 Disease Analysis (Dead Heart)",
        insectAnalysisTitle: "🐛 Insect Analysis",
        finalDiagnosisLabel: "Final Diagnosis:",
        visualDetectionLabel: "Visual Detection (Image):",
        symptomAnalysisLabel: "Symptom Analysis (TabNet):",
        fusedCertaintyLabel: "Fused Certainty:",
        recommendationTitle: "💡 Recommendation", // Added for enhancement
        // Dynamic Questions using the short keys from model_handler.py
        dynamic_questions: {
            "dh_q1": "Have you seen the central growing point of the stalk damaged or dead?",
            "dh_q2": "Is the dead central shoot straw-coloured?",
            "dh_q3": "After pulling the dead-heart, do you find maggots that look like secondary feeders (not primary borers)?",
            "dh_q4": "After pulling the dead-heart, do you find fresh live larvae inside the affected stem?",
            "dh_q5": "Is most of the visible damage inside the stem rather than outside?",
            "dh_q6": "Have you noticed insect attack when the leaves are still developing and soft?",
            "dh_q7": "Was the crop planted within the last 15 days?",
            "dh_q8": "Have you never seen moths flying during daytime?",
            "dh_q9": "Have you observed mating or egg-laying activity mostly at night?",
            "dh_q10": "Were any biological control insects released in the field?",
            "dh_q11": "Have you seen fully grown moths that are straw to light brown in colour?",
            "dh_q12": "Is the central shoot of young plants dry, brown, or straw-colored?",
            "dh_q13": "Does the central shoot come out easily when pulled gently?",
            "dh_q14": "Does the pulled shoot emit a foul or rotten odor?",
            "dh_q15": "Are leaves around the central shoot yellowing, wilting, or drying?",
            "dh_q16": "Do patches of plants show multiple stalks with dried or dead centers?",
            "dh_q17": "Has the percentage of dead hearts increased after recent rains or waterlogging?",
            "dh_q18": "Are affected plants stunted or shorter than surrounding healthy plants?",
            "dh_q19": "Do affected plants fail to produce new green shoots or leaves?",
            "dh_q20": "Are there soft, hollow, or tunnel-like areas inside the affected stalks?",
            "dh_q21": "Have you seen bunchy or abnormal growth at the top of affected stalks?",
            "dh_q22": "Are soil moisture and drainage poor in areas where dead hearts appear?",
            "dh_q23": "Are there no dry central shoots in the field?",
            "dh_q24": "Is plant height uniform and normal throughout the field?",
            "dh_q25": "When pulling the central shoot, is it firmly attached without coming out easily?",
            "dh_q26": "Does the shoot base smell fresh with no rotting or foul odor?",
            "dh_q27": "Are leaves healthy, green, and not wilting near the central shoot?",
            "dh_q28": "Do you have no patches with multiple dead or dried shoots?",
            "dh_q29": "Have symptoms decreased after improved irrigation or fertilization?",
            "dh_q30": "Is there no recurrence of dead heart symptoms from previous seasons?",
            "in_q1": "Is the crop ≤ 120 days old (i.e., within first 4 months)?",
            "in_q2": "Did the damage start after 4 months from planting?",
            "in_q3": "Did you first notice the damage between March and June?",
            "in_q4": "Did you first notice the damage between June and December?",
            "in_q5": "Did attacks start very soon after planting (within about 15 days)?",
            "in_q6": "Did the damage stop appearing after 4 months from planting?",
            "in_q7": "Is the peak damage appearing around 7–9 months after planting?",
            "in_q8": "Does the damage seem to start from the lower part of the stalk?",
            "in_q9": "Does the damage seem to start from the upper part of the stalk?",
            "in_q10": "Are bore holes within 15 cm from the soil?",
            "in_q11": "Are bore holes on the upper internodes?",
            "in_q12": "Are some stalk internodes malformed or constricted?",
            "in_q13": "Do you see small aerial roots appearing on the stalks above the ground?",
            "in_q14": "When you pull the dead central shoot, does it come out easily?",
            "in_q15": "Does the pulled shoot have a foul or bad smell?",
            "in_q16": "Is the pulled shoot difficult to remove?",
            "in_q17": "When removed, is there no bad smell from the shoot?",
            "in_q18": "Are insect eggs present on the underside of the lower leaves?",
            "in_q19": "Are insect eggs present on the underside of the top leaves?",
            "in_q20": "Are the eggs flat, white, and smaller than 1 mm?",
            "in_q21": "Does the larva have 5 visible stripes along its body?",
            "in_q22": "Does the larva have only 4 visible stripes along its body?",
            "in_q23": "Is fresh powder-like excreta seen near the base of the stalk?",
            "in_q24": "Is fresh excreta seen on the upper internodes?",
            "in_q25": "In the early stages, do you see only green leaf scraping without bore holes?",
            "in_q26": "Has the damage occurred only after internodes have fully developed?",
            "in_q27": "Along with dead shoots, do you see bunchy or abnormal top growth?",
            "in_q28": "Did you apply a high dose of nitrogen/urea before the damage started?",
            "in_q29": "Was trash mulching done early in the crop stage?",
            "in_q30": "Was earthing-up done to cover the lower stalk area?"
        }
    },
    ta: {
        // Static UI Text
        headerTitle: "🧑‍🌾 அக்ரி-AI கண்டறிதல்",
        headerLead: "முழுமையான பகுப்பாய்விற்கு பயிர் அல்லது பூச்சி படத்தை பதிவேற்றவும்.",
        translateButton: "மொழிபெயர்",
        step1Title: "படி 1: படத்தை பதிவேற்றவும்",
        uploadLabel: "🌿 படத்தை பதிவேற்றவும் (பயிர் அல்லது பூச்சி)",
        analyzeButton: "படத்தை பகுப்பாய்வு செய்யவும்",
        loadingSpinner: "ஏற்றுகிறது...",
        step2Title: "படி 2: சில கேள்விகளுக்கு பதிலளிக்கவும்",
        step2Lead: "கீழேயுள்ள படம் மற்றும் உங்கள் அவதானிப்புகளின் அடிப்படையில் கூடுதல் தகவல்களை வழங்கவும்.",
        getDiagnosisButton: "இறுதி நோய் கண்டறிதலைப் பெறுக",
        resultsTitle: "இறுதி பகுப்பாய்வு",
        resetButton: "மற்றொரு பயிரை பகுப்பாய்வு செய்யவும்",
        diseaseSectionTitle: "இறந்த இதய அறிகுறிகள்",
        insectSectionTitle: "லார்வா அறிகுறிகள்",
        yes: "ஆம்",
        no: "இல்லை",
        diseaseAnalysisTitle: "🌿 நோய் பகுப்பாய்வு ( இறந்த இதயம்)",
        insectAnalysisTitle: "🐛 பூச்சி பகுப்பாய்வு",
        finalDiagnosisLabel: "இறுதி நோய் கண்டறிதல்:",
        visualDetectionLabel: "காட்சி கண்டறிதல் (படம்):",
        symptomAnalysisLabel: "அறிகுறி பகுப்பாய்வு (டாப்நெட்):",
        fusedCertaintyLabel: "இணைந்த உறுதி:",
        recommendationTitle: "💡 பரிந்துரை",
        // Dynamic Questions (Translated to Tamil)
        dynamic_questions: {
            "dh_q1": "தண்டின் மைய வளர்ச்சிப் புள்ளி சேதமடைந்ததையோ அல்லது இறந்ததையோ பார்த்திருக்கிறீர்களா?",
            "dh_q2": "இறந்த மையத் தண்டு வைக்கோல் நிறத்தில் உள்ளதா?",
            "dh_q3": "இறந்த இதயத்தை இழுத்த பிறகு, இரண்டாம் நிலை உண்ணிகளாகத் தோற்றமளிக்கும் புழுக்களை (முதன்மை துளைப்பான்கள் அல்ல) காண்கிறீர்களா?",
            "dh_q4": "இறந்த இதயத்தை இழுத்த பிறகு, பாதிக்கப்பட்ட தண்டுக்குள் புதிய உயிருள்ள புழுக்களைக் காண்கிறீர்களா?",
            "dh_q5": "வெளிப்பக்கத்தை விட தண்டுக்குள் தான் அதிக சேதம் தெரிகிறதா?",
            "dh_q6": "இலைகள் இன்னும் வளர்ந்து மென்மையாக இருக்கும்போது பூச்சி தாக்குதலைக் கவனித்திருக்கிறீர்களா?",
            "dh_q7": "கடந்த 15 நாட்களுக்குள் பயிர் நடப்பட்டதா?",
            "dh_q8": "பகல் நேரத்தில் அந்துப்பூச்சிகள் பறப்பதை நீங்கள் பார்த்ததே இல்லையா?",
            "dh_q9": "பெரும்பாலும் இரவில் இனச்சேர்க்கை அல்லது முட்டையிடும் செயலைக் கவனித்திருக்கிறீர்களா?",
            "dh_q10": "வயலில் ஏதேனும் உயிரியல் கட்டுப்பாட்டு பூச்சிகள் விடப்பட்டதா?",
            "dh_q11": "வைக்கோல் முதல் வெளிர் பழுப்பு நிறத்தில் முழுமையாக வளர்ந்த அந்துப்பூச்சிகளைப் பார்த்திருக்கிறீர்களா?",
            "dh_q12": "இளம் தாவரங்களின் மையத் தண்டு காய்ந்த, பழுப்பு அல்லது வைக்கோல் நிறத்தில் உள்ளதா?",
            "dh_q13": "மெதுவாக இழுக்கும்போது மையத் தண்டு எளிதில் வெளியே வருகிறதா?",
            "dh_q14": "இழுக்கப்பட்ட தண்டு துர்நாற்றம் அல்லது அழுகிய வாசனையை வெளியிடுகிறதா?",
            "dh_q15": "மையத் தண்டைச் சுற்றியுள்ள இலைகள் மஞ்சள், வாடல் அல்லது காய்ந்து போகின்றனவா?",
            "dh_q16": "தாவரங்களின் திட்டுகளில் காய்ந்த அல்லது இறந்த மையங்களைக் கொண்ட பல தண்டுகள் தென்படுகின்றனவா?",
            "dh_q17": "சமீபத்திய மழை அல்லது நீர் தேக்கத்திற்குப் பிறகு இறந்த இதயங்களின் சதவீதம் அதிகரித்துள்ளதா?",
            "dh_q18": "பாதிக்கப்பட்ட தாவரங்கள் சுற்றியுள்ள ஆரோக்கியமான தாவரங்களை விட குட்டையாகவோ அல்லது வளர்ச்சிக் குன்றியோ உள்ளதா?",
            "dh_q19": "பாதிக்கப்பட்ட தாவரங்கள் புதிய பச்சை தளிர்கள் அல்லது இலைகளை உருவாக்கத் தவறுகின்றனவா?",
            "dh_q20": "பாதிக்கப்பட்ட தண்டுகளுக்குள் மென்மையான, உள்ளீடற்ற அல்லது சுரங்கம் போன்ற பகுதிகள் உள்ளதா?",
            "dh_q21": "பாதிக்கப்பட்ட தண்டுகளின் உச்சியில் கொத்தான அல்லது அசாதாரண வளர்ச்சியைக் கண்டிருக்கிறீர்களா?",
            "dh_q22": "இறந்த இதயங்கள் தோன்றும் பகுதிகளில் மண் ஈரம் மற்றும் வடிகால் மோசமாக உள்ளதா?",
            "dh_q23": "வயலில் காய்ந்த மையத் தண்டுகள் எதுவும் இல்லையா?",
            "dh_q24": "வயல் முழுவதும் தாவரத்தின் உயரம் சீராகவும் சாதாரணமாகவும் உள்ளதா?",
            "dh_q25": "மையத் தண்டை இழுக்கும்போது, அது எளிதில் வெளியே வராமல் உறுதியாக இணைக்கப்பட்டுள்ளதா?",
            "dh_q26": "தண்டின் அடிப்பகுதி அழுகல் அல்லது துர்நாற்றம் இல்லாமல் తాజాగా மணக்கிறதா?",
            "dh_q27": "மையத் தண்டுக்கு அருகில் உள்ள இலைகள் ஆரோக்கியமாகவும், பச்சையாகவும், வாடாமலும் உள்ளதா?",
            "dh_q28": "பல இறந்த அல்லது காய்ந்த தளிர்கள் கொண்ட திட்டுகள் உங்களிடம் இல்லையா?",
            "dh_q29": "மேம்படுத்தப்பட்ட நீர்ப்பாசனம் அல்லது உரமிட்ட பிறகு அறிகுறிகள் குறைந்துவிட்டதா?",
            "dh_q30": "முந்தைய பருவங்களிலிருந்து இறந்த இதய அறிகுறிகள் மீண்டும் ஏற்படவில்லையா?",
            "in_q1": "பயிர் 120 நாட்கள் அல்லது அதற்கும் குறைவாக உள்ளதா (அதாவது, முதல் 4 மாதங்களுக்குள்)?",
            "in_q2": "நட்ட 4 மாதங்களுக்குப் பிறகு சேதம் தொடங்கியதா?",
            "in_q3": "மார்ச் மற்றும் ஜூன் மாதங்களுக்கு இடையில் சேதத்தை முதலில் கவனித்தீர்களா?",
            "in_q4": "ஜூன் மற்றும் டிசம்பர் மாதங்களுக்கு இடையில் சேதத்தை முதலில் கவனித்தீர்களா?",
            "in_q5": "நட்ட உடனேயே (சுமார் 15 நாட்களுக்குள்) தாக்குதல் தொடங்கியதா?",
            "in_q6": "நட்ட 4 மாதங்களுக்குப் பிறகு சேதம் தோன்றுவது நின்றதா?",
            "in_q7": "நட்ட 7-9 மாதங்களில் உச்ச சேதம் தோன்றுகிறதா?",
            "in_q8": "தண்டின் கீழ்பகுதியிலிருந்து சேதம் தொடங்குவது போல் தெரிகிறதா?",
            "in_q9": "தண்டின் ಮೇಲ್ப் பகுதியிலிருந்து சேதம் தொடங்குவது போல் தெரிகிறதா?",
            "in_q10": "துளைகள் மண்ணிலிருந்து 15 செ.மீ.க்குள் உள்ளதா?",
            "in_q11": "துளைகள் மேல் கணுக்களில் உள்ளதா?",
            "in_q12": "சில தண்டு கணுக்கள் உருக்குலைந்து அல்லது சுருங்கி உள்ளதா?",
            "in_q13": "தரைக்கு மேலே தண்டுகளில் சிறிய வான்வழி வேர்கள் தோன்றுவதைப் பார்க்கிறீர்களா?",
            "in_q14": "இறந்த மையத் தண்டை இழுக்கும்போது, அது எளிதாக வருகிறதா?",
            "in_q15": "இழுக்கப்பட்ட தளிரில் துர்நாற்றம் வீசுகிறதா?",
            "in_q16": "இழுக்கப்பட்ட தண்டை அகற்றுவது கடினமாக உள்ளதா?",
            "in_q17": "அகற்றும்போது, தளிரிலிருந்து துர்நாற்றம் வரவில்லையா?",
            "in_q18": "பூச்சி முட்டைகள் கீழ் இலைகளின் அடிப்பக்கத்தில் உள்ளதா?",
            "in_q19": "பூச்சி முட்டைகள் மேல் இலைகளின் அடிப்பக்கத்தில் உள்ளதா?",
            "in_q20": "முட்டைகள் தட்டையாகவும், வெள்ளையாகவும், 1 மி.மீ.க்கும் குறைவாகவும் உள்ளதா?",
            "in_q21": "புழுவின் உடலில் 5 தெரியும் கோடுகள் உள்ளதா?",
            "in_q22": "புழுவின் உடலில் 4 தெரியும் கோடுகள் மட்டும் உள்ளதா?",
            "in_q23": "தண்டின் அருகே புதிய தூள் போன்ற கழிவுகள் காணப்படுகிறதா?",
            "in_q24": "மேல் கணுக்களில் புதிய கழிவுகள் காணப்படுகிறதா?",
            "in_q25": "ஆரம்ப கட்டங்களில், துளைகள் இல்லாமல் பச்சை இலை சுரண்டலை மட்டும் பார்க்கிறீர்களா?",
            "in_q26": "கணுக்கள் முழுமையாக வளர்ந்த பிறகுதான் சேதம் ஏற்பட்டுள்ளதா?",
            "in_q27": "இறந்த தளிர்களுடன், கொத்தான அல்லது அசாதாரணமான மேல் வளர்ச்சியைக் காண்கிறீர்களா?",
            "in_q28": "சேதம் தொடங்குவதற்கு முன் அதிக அளவு நைட்ரஜன்/யூரியா பயன்படுத்தினீர்களா?",
            "in_q29": "பயிர் பருவத்தின் ஆரம்பத்தில் குப்பை மூடாக்குதல் செய்யப்பட்டதா?",
            "in_q30": "கீழ் தண்டு பகுதியை மறைக்க மண் அணைப்பு செய்யப்பட்டதா?"
        }
    },
    hi: {
        // Static UI Text
        headerTitle: "🧑‍🌾 एग्री-एआई निदान",
        headerLead: "संपूर्ण विश्लेषण के लिए फसल या कीट की छवि अपलोड करें।",
        translateButton: "अनुवाद",
        step1Title: "चरण 1: चित्र अपलोड करें",
        uploadLabel: "🌿 चित्र अपलोड करें (फसल या कीट)",
        analyzeButton: "चित्र का विश्लेषण करें",
        loadingSpinner: "लोड हो रहा है...",
        step2Title: "चरण 2: कुछ प्रश्नों के उत्तर दें",
        step2Lead: "कृपया नीचे दी गई छवि और अपने अवलोकनों के आधार पर अतिरिक्त जानकारी प्रदान करें।",
        getDiagnosisButton: "अंतिम निदान प्राप्त करें",
        resultsTitle: "अंतिम विश्लेषण",
        resetButton: "दूसरी फसल का विश्लेषण करें",
        diseaseSectionTitle: "डेड-हार्ट के लक्षण",
        insectSectionTitle: "लार्वा के लक्षण",
        yes: "हाँ",
        no: "नहीं",
        diseaseAnalysisTitle: "🌿 रोग विश्लेषण (डेड हार्ट)",
        insectAnalysisTitle: "🐛 कीट विश्लेषण",
        finalDiagnosisLabel: "अंतिम निदान:",
        visualDetectionLabel: "दृश्य पहचान (छवि):",
        symptomAnalysisLabel: "लक्षण विश्लेषण (टैबनेट):",
        fusedCertaintyLabel: "संयुक्त निश्चितता:",
        recommendationTitle: "💡 सिफारिश",
        // Dynamic Questions (Translated to Hindi)
        dynamic_questions: {
            "dh_q1": "क्या आपने डंठल के केंद्रीय वृद्धि बिंदु को क्षतिग्रस्त या मृत देखा है?",
            "dh_q2": "क्या मृत केंद्रीय अंकुर भूसे के रंग का है?",
            "dh_q3": "डेड-हार्ट खींचने के बाद, क्या आपको द्वितीयक फीडर (प्राथमिक बोरर्स नहीं) जैसे मैगॉट मिलते हैं?",
            "dh_q4": "डेड-हार्ट खींचने के बाद, क्या आपको प्रभावित तने के अंदर ताज़े जीवित लार्वा मिलते हैं?",
            "dh_q5": "क्या अधिकांश दृश्य क्षति बाहर के बजाय तने के अंदर है?",
            "dh_q6": "क्या आपने पत्तियों के विकास और नरम होने पर कीटों का हमला देखा है?",
            "dh_q7": "क्या फसल पिछले 15 दिनों के भीतर लगाई गई थी?",
            "dh_q8": "क्या आपने दिन में पतंगों को उड़ते हुए कभी नहीं देखा है?",
            "dh_q9": "क्या आपने ज्यादातर रात में संभोग या अंडे देने की गतिविधि देखी है?",
            "dh_q10": "क्या खेत में कोई जैविक नियंत्रण कीड़े छोड़े गए थे?",
            "dh_q11": "क्या आपने पूरी तरह से विकसित पतंगे देखे हैं जो भूसे से हल्के भूरे रंग के होते हैं?",
            "dh_q12": "क्या युवा पौधों का केंद्रीय अंकुर सूखा, भूरा या भूसे के रंग का होता है?",
            "dh_q13": "क्या धीरे से खींचने पर केंद्रीय अंकुर आसानी से बाहर आ जाता है?",
            "dh_q14": "क्या खींचे गए अंकुर से दुर्गंध या सड़ी हुई गंध आती है?",
            "dh_q15": "क्या केंद्रीय अंकुर के आसपास की पत्तियाँ पीली पड़ रही हैं, मुरझा रही हैं या सूख रही हैं?",
            "dh_q16": "क्या पौधों के धब्बों में सूखे या मृत केंद्रों वाले कई डंठल दिखाई देते हैं?",
            "dh_q17": "क्या हाल ही में हुई बारिश या जलभराव के बाद डेड हार्ट का प्रतिशत बढ़ा है?",
            "dh_q18": "क्या प्रभावित पौधे आसपास के स्वस्थ पौधों की तुलना में अविकसित या छोटे हैं?",
            "dh_q19": "क्या प्रभावित पौधे नए हरे अंकुर या पत्तियां पैदा करने में विफल रहते हैं?",
            "dh_q20": "क्या प्रभावित डंठलों के अंदर नरम, खोखले या सुरंग जैसे क्षेत्र हैं?",
            "dh_q21": "क्या आपने प्रभावित डंठलों के शीर्ष पर गुच्छेदार या असामान्य वृद्धि देखी है?",
            "dh_q22": "क्या उन क्षेत्रों में मिट्टी की नमी और जल निकासी खराब है जहाँ डेड हार्ट दिखाई देते हैं?",
            "dh_q23": "क्या खेत में कोई सूखे केंद्रीय अंकुर नहीं हैं?",
            "dh_q24": "क्या पूरे खेत में पौधे की ऊंचाई एक समान और सामान्य है?",
            "dh_q25": "केंद्रीय अंकुर को खींचते समय, क्या यह आसानी से बाहर आए बिना मजबूती से जुड़ा होता है?",
            "dh_q26": "क्या अंकुर के आधार से सड़न या दुर्गंध के बिना ताज़ा गंध आती है?",
            "dh_q27": "क्या केंद्रीय अंकुर के पास की पत्तियाँ स्वस्थ, हरी और मुरझाई हुई नहीं हैं?",
            "dh_q28": "क्या आपके पास कई मृत या सूखे अंकुरों वाले कोई धब्बे नहीं हैं?",
            "dh_q29": "क्या बेहतर सिंचाई या उर्वरीकरण के बाद लक्षण कम हुए हैं?",
            "dh_q30": "क्या पिछले मौसमों से डेड हार्ट के लक्षणों की कोई पुनरावृत्ति नहीं हुई है?",
            "in_q1": "क्या फसल ≤ 120 दिन पुरानी है (यानी, पहले 4 महीनों के भीतर)?",
            "in_q2": "क्या क्षति रोपण के 4 महीने बाद शुरू हुई?",
            "in_q3": "क्या आपने पहली बार मार्च और जून के बीच क्षति देखी?",
            "in_q4": "क्या आपने पहली बार जून और दिसंबर के बीच क्षति देखी?",
            "in_q5": "क्या रोपण के तुरंत बाद (लगभग 15 दिनों के भीतर) हमले शुरू हुए?",
            "in_q6": "क्या रोपण के 4 महीने बाद क्षति दिखना बंद हो गई?",
            "in_q7": "क्या रोपण के 7-9 महीने के आसपास चरम क्षति दिखाई दे रही है?",
            "in_q8": "क्या क्षति डंठल के निचले हिस्से से शुरू होती दिख रही है?",
            "in_q9": "क्या क्षति डंठल के ऊपरी हिस्से से शुरू होती दिख रही है?",
            "in_q10": "क्या बोर होल मिट्टी से 15 सेमी के भीतर हैं?",
            "in_q11": "क्या बोर होल ऊपरी पोरों पर हैं?",
            "in_q12": "क्या कुछ डंठल के पोर विकृत या संकुचित हैं?",
            "in_q13": "क्या आप जमीन के ऊपर डंठलों पर छोटी हवाई जड़ें दिखाई दे रही हैं?",
            "in_q14": "जब आप मृत केंद्रीय अंकुर को खींचते हैं, तो क्या वह आसानी से बाहर आ जाता है?",
            "in_q15": "क्या खींचे गए अंकुर से दुर्गंध या खराब गंध आती है?",
            "in_q16": "क्या खींचे गए अंकुर को हटाना मुश्किल है?",
            "in_q17": "हटाने पर, क्या अंकुर से कोई खराब गंध नहीं आती है?",
            "in_q18": "क्या कीट के अंडे निचली पत्तियों के नीचे मौजूद हैं?",
            "in_q19": "क्या कीट के अंडे ऊपरी पत्तियों के नीचे मौजूद हैं?",
            "in_q20": "क्या अंडे सपाट, सफेद और 1 मिमी से छोटे हैं?",
            "in_q21": "क्या लार्वा के शरीर पर 5 दृश्य धारियाँ हैं?",
            "in_q22": "क्या लार्वा के शरीर पर केवल 4 दृश्य धारियाँ हैं?",
            "in_q23": "क्या डंठल के आधार के पास ताज़ा पाउडर जैसा मल दिखाई देता है?",
            "in_q24": "क्या ऊपरी पोरों पर ताज़ा मल दिखाई देता है?",
            "in_q25": "प्रारंभिक अवस्था में, क्या आप बोर होल के बिना केवल हरी पत्ती की खुरचन देखते हैं?",
            "in_q26": "क्या पोरों के पूरी तरह से विकसित होने के बाद ही क्षति हुई है?",
            "in_q27": "मृत अंकुरों के साथ, क्या आप गुच्छेदार या असामान्य शीर्ष वृद्धि देखते हैं?",
            "in_q28": "क्या आपने क्षति शुरू होने से पहले नाइट्रोजन/यूरिया की उच्च खुराक दी थी?",
            "in_q29": "क्या फसल की प्रारंभिक अवस्था में कचरा मल्चिंग किया गया था?",
            "in_q30": "क्या निचले डंठल क्षेत्र को ढकने के लिए मिट्टी चढ़ाई गई थी?"
        }
    }
};

function translatePage() {
    document.querySelectorAll('[data-translate-key]').forEach(el => {
        const key = el.getAttribute('data-translate-key');
        if (translations[currentLanguage] && translations[currentLanguage][key]) {
            el.innerText = translations[currentLanguage][key];
        }
    });

    // Re-render questions if the symptom card is visible
    if (symptomCard.style.display === 'block' && analysisData.questions) {
        displayQuestions(analysisData);
    }
    
    // Re-render results if the results card is visible
    if (resultsCard.style.display === 'block' && analysisData.results) {
        displayResults(analysisData.results);
    }
}

function setLanguage(lang) {
    currentLanguage = lang;
    translatePage();
}

// ============== EXISTING CODE (No changes needed below this line, except for displayResults) ==============
const uploadForm = document.getElementById('upload-form');
const imageInput = document.getElementById('image-input');
const spinner = document.getElementById('spinner');
const resetBtn = document.getElementById('reset-btn');
const uploadCard = document.getElementById('upload-card');
const symptomCard = document.getElementById('symptom-card');
const symptomForm = document.getElementById('symptom-form');
const submitSymptomsBtn = document.getElementById('submit-symptoms');
const resultsCard = document.getElementById('results-card');
const resultsContent = document.getElementById('results-content');
const uploadedImagePreview = document.getElementById('uploaded-image-preview');
const imageTypeText = document.querySelector('#symptom-card .text-muted');

let analysisData = {};

uploadForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    if (!imageInput.files[0]) { alert("Please select an image."); return; }
    spinner.style.display = 'block';
    const formData = new FormData();
    formData.append('image_file', imageInput.files[0]);
    try {
        const response = await fetch('/analyze', { method: 'POST', body: formData });
        if (!response.ok) throw new Error('Server error during image analysis.');
        analysisData = await response.json();
        uploadCard.style.display = 'none';
        uploadedImagePreview.src = analysisData.image_url;
        uploadedImagePreview.style.display = 'block';
        imageTypeText.textContent = analysisData.image_content_type;
        displayQuestions(analysisData);
        symptomCard.style.display = 'block';
    } catch (error) {
        console.error('Fetch Error:', error);
        alert('Could not connect to the server.');
    } finally {
        spinner.style.display = 'none';
    }
});

submitSymptomsBtn.addEventListener('click', async () => {
    // Correctly collect answers based on the unique name generated
    const diseaseAnswers = [];
    document.querySelectorAll('[name^="disease_"]').forEach(radio => {
        if (radio.checked && radio.value === 'yes') {
            diseaseAnswers.push('yes');
        } else if (radio.checked && radio.value === 'no') {
            diseaseAnswers.push('no');
        }
    });

    const insectAnswers = [];
    document.querySelectorAll('[name^="insect_"]').forEach(radio => {
        if (radio.checked && radio.value === 'yes') {
            insectAnswers.push('yes');
        } else if (radio.checked && radio.value === 'no') {
            insectAnswers.push('no');
        }
    });

    const payload = {
        yolo_disease_output: analysisData.yolo_disease_output,
        yolo_insect_output: analysisData.yolo_insect_output,
        disease_answers: diseaseAnswers,
        insect_answers: insectAnswers
    };
    try {
        const response = await fetch('/fuse', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        if (!response.ok) throw new Error('Server error during fusion.');
        const finalResults = await response.json();
        analysisData.results = finalResults;
        symptomCard.style.display = 'none';
        displayResults(finalResults);
        resultsCard.style.display = 'block';
    } catch (error) {
        console.error('Fetch Error:', error);
        alert('Could not get final diagnosis.');
    }
});


resetBtn.addEventListener('click', () => { window.location.reload(); });

function displayQuestions(data) {
    symptomForm.innerHTML = ''; // Clear previous questions
    let questionCount = 0;

    // Helper to create a section title that can be translated
    const createSectionTitle = (key) => {
        const h6 = document.createElement('h6');
        h6.setAttribute('data-translate-key', key);
        h6.innerText = translations[currentLanguage][key];
        return h6;
    };

    if (data.questions && data.questions.disease_questions) {
        const diseaseSection = document.createElement('div');
        diseaseSection.className = 'mb-4';
        diseaseSection.appendChild(createSectionTitle('diseaseSectionTitle'));
        data.questions.disease_questions.forEach(q => diseaseSection.appendChild(createQuestionElement(q, 'disease')));
        symptomForm.appendChild(diseaseSection);
        questionCount += data.questions.disease_questions.length;
    }
    if (data.questions && data.questions.insect_questions) {
        const insectSection = document.createElement('div');
        insectSection.className = 'mb-4';
        insectSection.appendChild(createSectionTitle('insectSectionTitle'));
        data.questions.insect_questions.forEach(q => insectSection.appendChild(createQuestionElement(q, 'insect')));
        symptomForm.appendChild(insectSection);
        questionCount += data.questions.insect_questions.length;
    }
    submitSymptomsBtn.style.display = questionCount > 0 ? 'block' : 'none';
}


function createQuestionElement(q, type) {
    const uniqueId = q.key; // Use the key from the backend (e.g., "dh_q1")
    const formGroup = document.createElement('div');
    formGroup.className = 'mb-3';
    
    // Look up the translated question text using the key
    const questionText = translations[currentLanguage].dynamic_questions[q.key] || q.text;

    formGroup.innerHTML = `
        <label class="form-label small">${questionText}</label>
        <div>
            <input type="radio" class="btn-check" name="${type}_${uniqueId}" id="${uniqueId}-yes" value="yes" autocomplete="off">
            <label class="btn btn-outline-success btn-sm" for="${uniqueId}-yes" data-translate-key="yes">${translations[currentLanguage].yes}</label>
            
            <input type="radio" class="btn-check" name="${type}_${uniqueId}" id="${uniqueId}-no" value="no" autocomplete="off" checked>
            <label class="btn btn-outline-danger btn-sm" for="${uniqueId}-no" data-translate-key="no">${translations[currentLanguage].no}</label>
        </div>
    `;
    return formGroup;
}


// --- ENHANCED displayResults function ---
function displayResults(data) {
    const lang = translations[currentLanguage];
    resultsContent.innerHTML = ''; // Clear previous results

    const getConfidenceClass = (diag) => {
        if (!diag) return "text-muted";
        const lowerDiag = diag.toLowerCase();
        if (lowerDiag.includes("present") || lowerDiag.includes("borer") || lowerDiag.includes("unconfirmed")) {
            return "text-danger fw-bold";
        }
        if (lowerDiag.includes("healthy") || lowerDiag.includes("not present")) {
            return "text-success fw-bold";
        }
        return "text-dark";
    };

    const analysisResult = data.result;
    const analysisType = data.analysis_type;

    let title, finalDiagnosis, visual, symptom, fused, recommendation;

    if (analysisType === 'disease') {
        title = lang.diseaseAnalysisTitle;
        finalDiagnosis = analysisResult.final_diagnosis;
        visual = analysisResult.yolo_output;
        symptom = `Prob. ${analysisResult.tabnet_probability}`;
        fused = `Prob. ${analysisResult.fused_probability}`;
        recommendation = analysisResult.recommendation ? analysisResult.recommendation[currentLanguage] : "No specific recommendation available.";
    } else if (analysisType === 'insect') {
        title = lang.insectAnalysisTitle;
        finalDiagnosis = analysisResult.final_diagnosis;
        visual = analysisResult.yolo_output;
        symptom = analysisResult.tabnet_classification;
        fused = `Prob. ${analysisResult.fused_probability}`;
        recommendation = analysisResult.recommendation ? analysisResult.recommendation[currentLanguage] : "No specific recommendation available.";
    }

    resultsContent.innerHTML = `
        <div class="mb-4">
            <h6>${title}</h6>
            <p class="mb-1"><strong>${lang.finalDiagnosisLabel}</strong> <span class="${getConfidenceClass(finalDiagnosis)}">${finalDiagnosis}</span></p>
            <ul class="list-unstyled small text-muted mb-3">
                <li>${lang.visualDetectionLabel} ${visual}</li>
                <li>${lang.symptomAnalysisLabel} ${symptom}</li>
                <li>${lang.fusedCertaintyLabel} ${fused}</li>
            </ul>
        </div>
        <div>
            <h6 data-translate-key="recommendationTitle">${lang.recommendationTitle}</h6>
            <p class="small">${recommendation}</p>
        </div>
    `;
}


document.addEventListener('DOMContentLoaded', () => { setLanguage('en'); });