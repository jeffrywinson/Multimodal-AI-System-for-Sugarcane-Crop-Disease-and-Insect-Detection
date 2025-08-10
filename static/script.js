// static/script.js

// ============== TRANSLATION SETUP ==============
let currentLanguage = 'en';

// --- Comprehensive translation object including all dynamic questions ---
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
        // Dynamic Questions (Your full English to English mapping)
        dynamic_questions: {
            "Have you seen the central growing point of the stalk damaged or dead?": "Have you seen the central growing point of the stalk damaged or dead?",
            "Is the dead central shoot straw-coloured?": "Is the dead central shoot straw-coloured?",
            "After pulling the dead-heart, do you find maggots that look like secondary feeders (not primary borers)?": "After pulling the dead-heart, do you find maggots that look like secondary feeders (not primary borers)?",
            "After pulling the dead-heart, do you find fresh live larvae inside the affected stem?": "After pulling the dead-heart, do you find fresh live larvae inside the affected stem?",
            "Is most of the visible damage inside the stem rather than outside?": "Is most of the visible damage inside the stem rather than outside?",
            "Have you noticed insect attack when the leaves are still developing and soft?": "Have you noticed insect attack when the leaves are still developing and soft?",
            "Was the crop planted within the last 15 days?": "Was the crop planted within the last 15 days?",
            "Have you never seen moths flying during daytime?": "Have you never seen moths flying during daytime?",
            "Have you observed mating or egg-laying activity mostly at night?": "Have you observed mating or egg-laying activity mostly at night?",
            "Were any biological control insects released in the field?": "Were any biological control insects released in the field?",
            "Have you seen fully grown moths that are straw to light brown in colour?": "Have you seen fully grown moths that are straw to light brown in colour?",
            "Is the central shoot of young plants dry, brown, or straw-colored?": "Is the central shoot of young plants dry, brown, or straw-colored?",
            "Does the central shoot come out easily when pulled gently?": "Does the central shoot come out easily when pulled gently?",
            "Does the pulled shoot emit a foul or rotten odor?": "Does the pulled shoot emit a foul or rotten odor?",
            "Are leaves around the central shoot yellowing, wilting, or drying?": "Are leaves around the central shoot yellowing, wilting, or drying?",
            "Do patches of plants show multiple stalks with dried or dead centers?": "Do patches of plants show multiple stalks with dried or dead centers?",
            "Has the percentage of dead hearts increased after recent rains or waterlogging?": "Has the percentage of dead hearts increased after recent rains or waterlogging?",
            "Are affected plants stunted or shorter than surrounding healthy plants?": "Are affected plants stunted or shorter than surrounding healthy plants?",
            "Do affected plants fail to produce new green shoots or leaves?": "Do affected plants fail to produce new green shoots or leaves?",
            "Are there soft, hollow, or tunnel-like areas inside the affected stalks?": "Are there soft, hollow, or tunnel-like areas inside the affected stalks?",
            "Have you seen bunchy or abnormal growth at the top of affected stalks?": "Have you seen bunchy or abnormal growth at the top of affected stalks?",
            "Are soil moisture and drainage poor in areas where dead hearts appear?": "Are soil moisture and drainage poor in areas where dead hearts appear?",
            "Are there no dry central shoots in the field?": "Are there no dry central shoots in the field?",
            "Is plant height uniform and normal throughout the field?": "Is plant height uniform and normal throughout the field?",
            "When pulling the central shoot, is it firmly attached without coming out easily?": "When pulling the central shoot, is it firmly attached without coming out easily?",
            "Does the shoot base smell fresh with no rotting or foul odor?": "Does the shoot base smell fresh with no rotting or foul odor?",
            "Are leaves healthy, green, and not wilting near the central shoot?": "Are leaves healthy, green, and not wilting near the central shoot?",
            "Do you have no patches with multiple dead or dried shoots?": "Do you have no patches with multiple dead or dried shoots?",
            "Have symptoms decreased after improved irrigation or fertilization?": "Have symptoms decreased after improved irrigation or fertilization?",
            "Is there no recurrence of dead heart symptoms from previous seasons?": "Is there no recurrence of dead heart symptoms from previous seasons?"
        }
    },
    ta: {
        // Static UI Text
        headerTitle: "🧑‍🌾 அக்ரி-AI கண்டறிதல்",
        headerLead: "முழுமையான பகுப்பாய்விற்கு பயிர் அல்லது பூச்சி படத்தை பதிவேற்றவும்.",
        translateButton: "மொழிபெயர்",
        step1Title: "படி 1: படத்தை பதிவேற்றவும்",
        uploadLabel: "🌿 படத்தை பதிவேற்றவும் (பயிர் அல்லது பூச்சி)",
        analyzeButton: "படத்தை பகுப்பாய்வு செய்யவும்",
        loadingSpinner: "ஏற்றுகிறது...",
        step2Title: "படி 2: சில கேள்விகளுக்கு பதிலளிக்கவும்",
        step2Lead: "கீழேயுள்ள படம் மற்றும் உங்கள் அவதானிப்புகளின் அடிப்படையில் கூடுதல் தகவல்களை வழங்கவும்.",
        getDiagnosisButton: "இறுதி நோய் கண்டறிதலைப் பெறுக",
        resultsTitle: "இறுதி பகுப்பாய்வு",
        resetButton: "மற்றொரு பயிரை பகுப்பாய்வு செய்யவும்",
        diseaseSectionTitle: "இறந்த இதய அறிகுறிகள்",
        insectSectionTitle: "லார்வா அறிகுறிகள்",
        yes: "ஆம்",
        no: "இல்லை",
        diseaseAnalysisTitle: "🌿 நோய் பகுப்பாய்வு ( இறந்த இதயம்)",
        insectAnalysisTitle: "🐛 பூச்சி பகுப்பாய்வு",
        finalDiagnosisLabel: "இறுதி நோய் கண்டறிதல்:",
        visualDetectionLabel: "காட்சி கண்டறிதல் (படம்):",
        symptomAnalysisLabel: "அறிகுறி பகுப்பாய்வு (டாப்நெட்):",
        fusedCertaintyLabel: "இணைந்த உறுதி:",
        // Dynamic Questions (Translated to Tamil)
        dynamic_questions: {
            "Have you seen the central growing point of the stalk damaged or dead?": "தண்டின் மைய வளர்ச்சிப் புள்ளி சேதமடைந்ததையோ அல்லது இறந்ததையோ பார்த்திருக்கிறீர்களா?",
            "Is the dead central shoot straw-coloured?": "இறந்த மையத் தண்டு வைக்கோல் நிறத்தில் உள்ளதா?",
            "After pulling the dead-heart, do you find maggots that look like secondary feeders (not primary borers)?": "இறந்த இதயத்தை இழுத்த பிறகு, இரண்டாம் நிலை உண்ணிகளாகத் தோற்றமளிக்கும் புழுக்களை (முதன்மை துளைப்பான்கள் அல்ல) காண்கிறீர்களா?",
            "After pulling the dead-heart, do you find fresh live larvae inside the affected stem?": "இறந்த இதயத்தை இழுத்த பிறகு, பாதிக்கப்பட்ட தண்டுக்குள் புதிய உயிருள்ள புழுக்களைக் காண்கிறீர்களா?",
            "Is most of the visible damage inside the stem rather than outside?": "வெளிப்பக்கத்தை விட தண்டுக்குள் தான் அதிக சேதம் தெரிகிறதா?",
            "Have you noticed insect attack when the leaves are still developing and soft?": "இலைகள் இன்னும் வளர்ந்து மென்மையாக இருக்கும்போது பூச்சி தாக்குதலைக் கவனித்திருக்கிறீர்களா?",
            "Was the crop planted within the last 15 days?": "கடந்த 15 நாட்களுக்குள் பயிர் நடப்பட்டதா?",
            "Have you never seen moths flying during daytime?": "பகல் நேரத்தில் அந்துப்பூச்சிகள் பறப்பதை நீங்கள் பார்த்ததே இல்லையா?",
            "Have you observed mating or egg-laying activity mostly at night?": "பெரும்பாலும் இரவில் இனச்சேர்க்கை அல்லது முட்டையிடும் செயலைக் கவனித்திருக்கிறீர்களா?",
            "Were any biological control insects released in the field?": "வயலில் ஏதேனும் உயிரியல் கட்டுப்பாட்டு பூச்சிகள் விடப்பட்டதா?",
            "Have you seen fully grown moths that are straw to light brown in colour?": "வைக்கோல் முதல் வெளிர் பழுப்பு நிறத்தில் முழுமையாக வளர்ந்த அந்துப்பூச்சிகளைப் பார்த்திருக்கிறீர்களா?",
            "Is the central shoot of young plants dry, brown, or straw-colored?": "இளம் தாவரங்களின் மையத் தண்டு காய்ந்த, பழுப்பு அல்லது வைக்கோல் நிறத்தில் உள்ளதா?",
            "Does the central shoot come out easily when pulled gently?": "மெதுவாக இழுக்கும்போது மையத் தண்டு எளிதில் வெளியே வருகிறதா?",
            "Does the pulled shoot emit a foul or rotten odor?": "இழுக்கப்பட்ட தண்டு துர்நாற்றம் அல்லது அழுகிய வாசனையை வெளியிடுகிறதா?",
            "Are leaves around the central shoot yellowing, wilting, or drying?": "மையத் தண்டைச் சுற்றியுள்ள இலைகள் மஞ்சள், வாடல் அல்லது காய்ந்து போகின்றனவா?",
            "Do patches of plants show multiple stalks with dried or dead centers?": "தாவரங்களின் திட்டுகளில் காய்ந்த அல்லது இறந்த மையங்களைக் கொண்ட பல தண்டுகள் தென்படுகின்றனவா?",
            "Has the percentage of dead hearts increased after recent rains or waterlogging?": "சமீபத்திய மழை அல்லது நீர் தேக்கத்திற்குப் பிறகு இறந்த இதயங்களின் சதவீதம் அதிகரித்துள்ளதா?",
            "Are affected plants stunted or shorter than surrounding healthy plants?": "பாதிக்கப்பட்ட தாவரங்கள் சுற்றியுள்ள ஆரோக்கியமான தாவரங்களை விட குட்டையாகவோ அல்லது வளர்ச்சிக் குன்றியோ உள்ளதா?",
            "Do affected plants fail to produce new green shoots or leaves?": "பாதிக்கப்பட்ட தாவரங்கள் புதிய பச்சை தளிர்கள் அல்லது இலைகளை உருவாக்கத் தவறுகின்றனவா?",
            "Are there soft, hollow, or tunnel-like areas inside the affected stalks?": "பாதிக்கப்பட்ட தண்டுகளுக்குள் மென்மையான, உள்ளீடற்ற அல்லது சுரங்கம் போன்ற பகுதிகள் உள்ளதா?",
            "Have you seen bunchy or abnormal growth at the top of affected stalks?": "பாதிக்கப்பட்ட தண்டுகளின் உச்சியில் கொத்தான அல்லது அசாதாரண வளர்ச்சியைக் கண்டிருக்கிறீர்களா?",
            "Are soil moisture and drainage poor in areas where dead hearts appear?": "இறந்த இதயங்கள் தோன்றும் பகுதிகளில் மண் ஈரம் மற்றும் வடிகால் மோசமாக உள்ளதா?",
            "Are there no dry central shoots in the field?": "வயலில் காய்ந்த மையத் தண்டுகள் எதுவும் இல்லையா?",
            "Is plant height uniform and normal throughout the field?": "வயல் முழுவதும் தாவரத்தின் உயரம் சீராகவும் சாதாரணமாகவும் உள்ளதா?",
            "When pulling the central shoot, is it firmly attached without coming out easily?": "மையத் தண்டை இழுக்கும்போது, அது எளிதில் வெளியே வராமல் உறுதியாக இணைக்கப்பட்டுள்ளதா?",
            "Does the shoot base smell fresh with no rotting or foul odor?": "தண்டின் அடிப்பகுதி அழுகல் அல்லது துர்நாற்றம் இல்லாமல் తాజాగా மணக்கிறதா?",
            "Are leaves healthy, green, and not wilting near the central shoot?": "மையத் தண்டுக்கு அருகில் உள்ள இலைகள் ஆரோக்கியமாகவும், பச்சையாகவும், வாடாமலும் உள்ளதா?",
            "Do you have no patches with multiple dead or dried shoots?": "பல இறந்த அல்லது காய்ந்த தளிர்கள் கொண்ட திட்டுகள் உங்களிடம் இல்லையா?",
            "Have symptoms decreased after improved irrigation or fertilization?": "மேம்படுத்தப்பட்ட நீர்ப்பாசனம் அல்லது உரமிட்ட பிறகு அறிகுறிகள் குறைந்துவிட்டதா?",
            "Is there no recurrence of dead heart symptoms from previous seasons?": "முந்தைய பருவங்களிலிருந்து இறந்த இதய அறிகுறிகள் மீண்டும் ஏற்படவில்லையா?"
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
        // Dynamic Questions (Translated to Hindi)
        dynamic_questions: {
            "Have you seen the central growing point of the stalk damaged or dead?": "क्या आपने डंठल के केंद्रीय वृद्धि बिंदु को क्षतिग्रस्त या मृत देखा है?",
            "Is the dead central shoot straw-coloured?": "क्या मृत केंद्रीय अंकुर भूसे के रंग का है?",
            "After pulling the dead-heart, do you find maggots that look like secondary feeders (not primary borers)?": "डेड-हार्ट खींचने के बाद, क्या आपको द्वितीयक फीडर (प्राथमिक बोरर्स नहीं) जैसे मैगॉट मिलते हैं?",
            "After pulling the dead-heart, do you find fresh live larvae inside the affected stem?": "डेड-हार्ट खींचने के बाद, क्या आपको प्रभावित तने के अंदर ताज़े जीवित लार्वा मिलते हैं?",
            "Is most of the visible damage inside the stem rather than outside?": "क्या अधिकांश दृश्य क्षति बाहर के बजाय तने के अंदर है?",
            "Have you noticed insect attack when the leaves are still developing and soft?": "क्या आपने पत्तियों के विकास और नरम होने पर कीटों का हमला देखा है?",
            "Was the crop planted within the last 15 days?": "क्या फसल पिछले 15 दिनों के भीतर लगाई गई थी?",
            "Have you never seen moths flying during daytime?": "क्या आपने दिन में पतंगों को उड़ते हुए कभी नहीं देखा है?",
            "Have you observed mating or egg-laying activity mostly at night?": "क्या आपने ज्यादातर रात में संभोग या अंडे देने की गतिविधि देखी है?",
            "Were any biological control insects released in the field?": "क्या खेत में कोई जैविक नियंत्रण कीड़े छोड़े गए थे?",
            "Have you seen fully grown moths that are straw to light brown in colour?": "क्या आपने पूरी तरह से विकसित पतंगे देखे हैं जो भूसे से हल्के भूरे रंग के होते हैं?",
            "Is the central shoot of young plants dry, brown, or straw-colored?": "क्या युवा पौधों का केंद्रीय अंकुर सूखा, भूरा या भूसे के रंग का होता है?",
            "Does the central shoot come out easily when pulled gently?": "क्या धीरे से खींचने पर केंद्रीय अंकुर आसानी से बाहर आ जाता है?",
            "Does the pulled shoot emit a foul or rotten odor?": "क्या खींचे गए अंकुर से दुर्गंध या सड़ी हुई गंध आती है?",
            "Are leaves around the central shoot yellowing, wilting, or drying?": "क्या केंद्रीय अंकुर के आसपास की पत्तियाँ पीली पड़ रही हैं, मुरझा रही हैं या सूख रही हैं?",
            "Do patches of plants show multiple stalks with dried or dead centers?": "क्या पौधों के धब्बों में सूखे या मृत केंद्रों वाले कई डंठल दिखाई देते हैं?",
            "Has the percentage of dead hearts increased after recent rains or waterlogging?": "क्या हाल ही में हुई बारिश या जलभराव के बाद डेड हार्ट का प्रतिशत बढ़ा है?",
            "Are affected plants stunted or shorter than surrounding healthy plants?": "क्या प्रभावित पौधे आसपास के स्वस्थ पौधों की तुलना में अविकसित या छोटे हैं?",
            "Do affected plants fail to produce new green shoots or leaves?": "क्या प्रभावित पौधे नए हरे अंकुर या पत्तियां पैदा करने में विफल रहते हैं?",
            "Are there soft, hollow, or tunnel-like areas inside the affected stalks?": "क्या प्रभावित डंठलों के अंदर नरम, खोखले या सुरंग जैसे क्षेत्र हैं?",
            "Have you seen bunchy or abnormal growth at the top of affected stalks?": "क्या आपने प्रभावित डंठलों के शीर्ष पर गुच्छेदार या असामान्य वृद्धि देखी है?",
            "Are soil moisture and drainage poor in areas where dead hearts appear?": "क्या उन क्षेत्रों में मिट्टी की नमी और जल निकासी खराब है जहाँ डेड हार्ट दिखाई देते हैं?",
            "Are there no dry central shoots in the field?": "क्या खेत में कोई सूखे केंद्रीय अंकुर नहीं हैं?",
            "Is plant height uniform and normal throughout the field?": "क्या पूरे खेत में पौधे की ऊंचाई एक समान और सामान्य है?",
            "When pulling the central shoot, is it firmly attached without coming out easily?": "केंद्रीय अंकुर को खींचते समय, क्या यह आसानी से बाहर आए बिना मजबूती से जुड़ा होता है?",
            "Does the shoot base smell fresh with no rotting or foul odor?": "क्या अंकुर के आधार से सड़न या दुर्गंध के बिना ताज़ा गंध आती है?",
            "Are leaves healthy, green, and not wilting near the central shoot?": "क्या केंद्रीय अंकुर के पास की पत्तियाँ स्वस्थ, हरी और मुरझाई हुई नहीं हैं?",
            "Do you have no patches with multiple dead or dried shoots?": "क्या आपके पास कई मृत या सूखे अंकुरों वाले कोई धब्बे नहीं हैं?",
            "Have symptoms decreased after improved irrigation or fertilization?": "क्या बेहतर सिंचाई या उर्वरीकरण के बाद लक्षण कम हुए हैं?",
            "Is there no recurrence of dead heart symptoms from previous seasons?": "क्या पिछले मौसमों से डेड हार्ट के लक्षणों की कोई पुनरावृत्ति नहीं हुई है?"
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
    if (resultsCard.style.display === 'block' && analysisData.results) {
        displayResults(analysisData.results);
    }
}

function setLanguage(lang) {
    currentLanguage = lang;
    translatePage();
}

// ============== EXISTING CODE (with minimal fixes) ==============
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
const imageTypeText = document.querySelector('#symptom-card .text-muted'); // More specific selector

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
    const diseaseAnswers = Array.from(document.querySelectorAll('[name^="disease_"]:checked')).map(input => input.value);
    const insectAnswers = Array.from(document.querySelectorAll('[name^="insect_"]:checked')).map(input => input.value);
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
    symptomForm.innerHTML = '';
    let questionCount = 0;
    if (data.questions && data.questions.disease_questions) {
        const diseaseSection = createSection('diseaseSectionTitle');
        data.questions.disease_questions.forEach((q, i) => diseaseSection.appendChild(createQuestionElement(q, 'disease_', i)));
        symptomForm.appendChild(diseaseSection);
        questionCount += data.questions.disease_questions.length;
    }
    if (data.questions && data.questions.insect_questions) {
        const insectSection = createSection('insectSectionTitle');
        data.questions.insect_questions.forEach((q, i) => insectSection.appendChild(createQuestionElement(q, 'insect_', i)));
        symptomForm.appendChild(insectSection);
        questionCount += data.questions.insect_questions.length;
    }
    submitSymptomsBtn.style.display = questionCount > 0 ? 'block' : 'none';
}

function createSection(titleKey) {
    const section = document.createElement('div');
    section.className = 'question-section';
    section.innerHTML = `<h6>${translations[currentLanguage][titleKey]}</h6>`;
    return section;
}

function createQuestionElement(q, prefix, index) {
    const uniqueId = prefix + q.key; // Use the key from the backend
    const formGroup = document.createElement('div');
    formGroup.className = 'mb-3';
    formGroup.innerHTML = `
        <label class="form-label small">${(translations[currentLanguage].dynamic_questions[q.key]) || q.text}</label>
        <div>
            <input type="radio" class="btn-check" name="${uniqueId}" id="${uniqueId}-yes" value="yes" autocomplete="off" required>
            <label class="btn btn-outline-success btn-sm" for="${uniqueId}-yes">${translations[currentLanguage].yes}</label>
            <input type="radio" class="btn-check" name="${uniqueId}" id="${uniqueId}-no" value="no" autocomplete="off" checked>
            <label class="btn btn-outline-danger btn-sm" for="${uniqueId}-no">${translations[currentLanguage].no}</label>
        </div>
    `;
    return formGroup;
}

// In script.js

function displayResults(data) {
    const lang = translations[currentLanguage];
    const resultsContent = document.getElementById('results-content');
    resultsContent.innerHTML = ''; // Clear previous results

    // Helper function to determine text color based on diagnosis
    const getConfidenceClass = (diag) => {
        if (!diag) return "text-muted";
        if (diag.toLowerCase().includes("present") || diag.toLowerCase().includes("borer") || diag.toLowerCase().includes("unconfirmed")) {
            return "text-danger fw-bold";
        }
        if (diag.toLowerCase().includes("healthy") || diag.toLowerCase().includes("not present")) {
            return "text-success fw-bold";
        }
        return "text-dark";
    };

    // Check the analysis type sent from the backend
    if (data.analysis_type === 'disease') {
        const disease = data.result;
        resultsContent.innerHTML = `
            <div class="mb-3">
                <h6>${lang.diseaseAnalysisTitle}</h6>
                <p class="mb-1"><strong>${lang.finalDiagnosisLabel}</strong> <span class="${getConfidenceClass(disease.final_diagnosis)}">${disease.final_diagnosis}</span></p>
                <ul class="list-unstyled small text-muted">
                    <li>${lang.visualDetectionLabel} ${disease.yolo_output}</li>
                    <li>${lang.symptomAnalysisLabel} Prob. ${disease.tabnet_probability}</li>
                    <li>${lang.fusedCertaintyLabel} Prob. ${disease.fused_probability}</li>
                </ul>
            </div>
        `;
    } else if (data.analysis_type === 'insect') {
        const insect = data.result;
        resultsContent.innerHTML = `
            <div>
                <h6>${lang.insectAnalysisTitle}</h6>
                <p class="mb-1"><strong>${lang.finalDiagnosisLabel}</strong> <span class="${getConfidenceClass(insect.final_diagnosis)}">${insect.final_diagnosis}</span></p>
                <ul class="list-unstyled small text-muted">
                    <li>${lang.visualDetectionLabel} ${insect.yolo_output}</li>
                    <li>${lang.symptomAnalysisLabel} ${insect.tabnet_classification}</li>
                    <li>${lang.fusedCertaintyLabel} Prob. ${insect.fused_probability}</li>
                </ul>
            </div>
        `;
    }
}


document.addEventListener('DOMContentLoaded', () => { translatePage(); });