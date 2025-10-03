// frontend/script.js - 完整最终版 (包含所有可选字段，包括“职业”)
document.addEventListener('DOMContentLoaded', () => {
    // --- 1. 状态管理 ---
    let personas = [];
    let currentPersonaIndex = -1;
    let productBase64Image = null;

    // --- 2. DOM 元素选择器 ---
    const pagePersona = document.getElementById('page-persona-creation');
    const pageProduct = document.getElementById('page-product-info');
    const pageResults = document.getElementById('page-results');
    const personaTabsContainer = document.getElementById('persona-tabs');
    const personaForm = document.getElementById('persona-form');


    // (必填字段)
    const genderEl = document.getElementById('gender');
    const ageEl = document.getElementById('age');
    const cityEl = document.getElementById('city');
    const mbtiEl = document.getElementById('mbti');
    
    // (可选字段)
    const professionEl = document.getElementById('profession');
    const educationEl = document.getElementById('education');
    const incomeEl = document.getElementById('income');
    const drinkFrequencyEl = document.getElementById('drink_frequency');
    const drinkingHistoryEl = document.getElementById('drinking_history');
    const expectedPriceEl = document.getElementById('expected_price');
    const preferredAromaEl = document.getElementById('preferred_aroma');
    
    // 按钮
    const btnAddPersona = document.getElementById('btn-add-persona');
    const btnSavePersona = document.getElementById('btn-save-persona');
    const btnNextPage = document.getElementById('btn-next-page');
    const btnBackToPersonas = document.getElementById('btn-back-to-personas');
    const btnSubmit = document.getElementById('btn-submit');
    const btnStartOver = document.getElementById('btn-start-over');

    // 产品信息表单
    const productDescriptionEl = document.getElementById('product-description');
    const productImageEl = document.getElementById('product-image');
    const imagePreview = document.getElementById('image-preview');

    // 结果展示区域
    const loadingIndicator = document.getElementById('loading-indicator');
    const resultsContent = document.getElementById('results-content');
    const summaryContainer = document.getElementById('summary-report-container');
    const individualContainer = document.getElementById('individual-reports-container');

    // --- 3. 核心功能函数 ---

    // 从表单读取所有字段值，整合成一个对象
    function getPersonaObjectFromForm() {
        return {
            gender: genderEl.value,
            age: ageEl.value,
            city: cityEl.value,
            mbti: mbtiEl.value,
            profession: professionEl.value,
            education: educationEl.value,
            income: incomeEl.value,
            drink_frequency: drinkFrequencyEl.value,
            drinking_history: drinkingHistoryEl.value,
            expected_price: expectedPriceEl.value,
            preferred_aroma: preferredAromaEl.value,
        };
    }

    // 将一个画像对象的数据加载到表单中
    function loadPersonaToForm(persona) {
        genderEl.value = persona.gender;
        ageEl.value = persona.age;
        cityEl.value = persona.city;
        mbtiEl.value = persona.mbti;
        professionEl.value = persona.profession || '';
        educationEl.value = persona.education || '';
        incomeEl.value = persona.income || '';
        drinkFrequencyEl.value = persona.drink_frequency || '';
        drinkingHistoryEl.value = persona.drinking_history || '';
        expectedPriceEl.value = persona.expected_price || '';
        preferredAromaEl.value = persona.preferred_aroma || '';
    }
    
    const navigateTo = (page) => {
        pagePersona.style.display = 'none';
        pageProduct.style.display = 'none';
        pageResults.style.display = 'none';
        page.style.display = 'block';
    };
    
    const setLoadingMessage = (message) => {
        const p = loadingIndicator.querySelector('p');
        if (p) p.textContent = message;
    }

    const renderTabs = () => {
        personaTabsContainer.innerHTML = '';
        personas.forEach((persona, index) => {
            const tab = document.createElement('div');
            tab.className = 'persona-tab';
            tab.classList.add(index === currentPersonaIndex ? 'active' : 'inactive');
            tab.textContent = `画像 ${index + 1}`;
            tab.addEventListener('click', () => loadPersonaForEditing(index));
            personaTabsContainer.appendChild(tab);
        });
    };

    const clearForm = () => {
        personaForm.reset();
        currentPersonaIndex = -1;
        btnSavePersona.style.display = 'none';
        btnAddPersona.style.display = 'block';
        renderTabs();
    };

    const loadPersonaForEditing = (index) => {
        currentPersonaIndex = index;
        const persona = personas[index];
        loadPersonaToForm(persona);

        btnAddPersona.style.display = 'none';
        btnSavePersona.style.display = 'block';
        renderTabs();
    };

    const addPersona = () => {
        const newPersona = getPersonaObjectFromForm();
        personas.push(newPersona);
        currentPersonaIndex = personas.length - 1;
        clearForm();
        currentPersonaIndex = personas.length - 1;
        renderTabs();
    };

    const savePersona = () => {
        if (currentPersonaIndex > -1) {
            personas[currentPersonaIndex] = getPersonaObjectFromForm();
            clearForm();
        }
    };
    
    const handleImageUpload = (event) => {
        const file = event.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (e) => {
                productBase64Image = e.target.result;
                imagePreview.src = e.target.result;
                imagePreview.style.display = 'block';
            };
            reader.readAsDataURL(file);
        }
    };
    
    const submitForReport = async () => {
        if (personas.length === 0) {
            alert('请至少添加一个画像。');
            return;
        }
        if (!productDescriptionEl.value || !productBase64Image) {
            alert('请提供产品描述和图片。');
            return;
        }

        navigateTo(pageResults);
        resultsContent.style.display = 'none';
        loadingIndicator.style.display = 'block';
        setLoadingMessage('正在生成报告... 页面将在此等待AI分析完成，此过程可能需要几分钟或更长时间。');

        const payload = {
            personas: personas,
            productData: {
                description: productDescriptionEl.value,
                image: productBase64Image,
            }
        };

        try {
            const response = await fetch('/generate_report', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });

            const results = await response.json();

            if (!response.ok) {
                throw new Error(results.error || `服务器响应状态: ${response.status}`);
            }

            loadingIndicator.style.display = 'none';
            resultsContent.style.display = 'block';
            displayResults(results);

        } catch (error) {
            loadingIndicator.style.display = 'block';
            resultsContent.style.display = 'none';
            setLoadingMessage(`发生错误: ${error.message}。请检查后端终端获取更多信息。`);
        }
    };

    const displayResults = (data) => {
        const genderMap = { 'Male': '男性', 'Female': '女性' };
        const cityMap = {
            'Beijing': '北京',
            'Shanghai': '上海',
            'Shenzhen': '深圳',
            'Guangzhou': '广州',
            'Chengdu': '成都'
        };

        summaryContainer.innerText = data.summary_report;
        individualContainer.innerHTML = '';
        
        data.individual_reports.forEach(item => {
            const card = document.createElement('div');
            card.className = 'report-card';
            const persona = item.persona_details;
            
            const translatedGender = genderMap[persona.gender] || persona.gender;
            const translatedCity = cityMap[persona.city] || persona.city;

            card.innerHTML = `
                <h3>画像 ${item.persona_id} 独立报告</h3>
                <p><strong>画像简介:</strong> ${persona.age}岁${translatedGender}，来自${translatedCity}，MBTI: ${persona.mbti}</p>
                <p>${item.report.trim()}</p>
            `;
            individualContainer.appendChild(card);
        });
    };

    // --- 4. 事件监听器 ---
    btnAddPersona.addEventListener('click', addPersona);
    btnSavePersona.addEventListener('click', savePersona);
    productImageEl.addEventListener('change', handleImageUpload);
    
    btnNextPage.addEventListener('click', () => {
        if (personas.length > 0) navigateTo(pageProduct);
        else alert('请至少添加一个画像再进行下一步。');
    });

    btnBackToPersonas.addEventListener('click', () => navigateTo(pagePersona));
    btnSubmit.addEventListener('click', submitForReport);
    btnStartOver.addEventListener('click', () => window.location.reload());
});