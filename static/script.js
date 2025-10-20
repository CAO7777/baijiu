// static/script.js (v4.7 - Editable Personas)

document.addEventListener('DOMContentLoaded', () => {
    // --- 1. STATE MANAGEMENT ---
    let personas = [];
    let personaFilePath = null;
    let personaSummaryStats = {};
    let individualReportsRendered = 0;
    let productImageDataURL = null;
    let availableCities = []; 
    // 💡 新增：用于跟踪正在编辑的画像
    let currentlyEditingPersonaIndex = null;

    // --- 2. DOM ELEMENT SELECTORS ---
    // ... (旧的选择器保持不变) ...
    const steps = document.querySelectorAll('.panel.step');
    const pills = document.querySelectorAll('.step-pill');
    const personaForm = document.getElementById('persona-form');
    const personaCountEl = document.getElementById('personaCount');
    const ageMinEl = document.getElementById('ageMin');
    const ageMaxEl = document.getElementById('ageMax');
    const cityRatioList = document.getElementById('cityRatioList');
    const addCityRatioBtn = document.getElementById('addCityRatio');
    const generatePersonasBtn = document.getElementById('generatePersonas');
    const personaSummaryEl = document.getElementById('personaSummary');
    const productDescEl = document.getElementById('productDesc');
    const productImgInput = document.getElementById('productImg');
    const productPreviewEl = document.getElementById('productPreview');
    const genReportBtn = document.getElementById('genReport');
    const individualContainer = document.getElementById('individual-reports-container');
    const nextToSummaryContainer = document.getElementById('next-to-summary-container');
    const summaryContainer = document.getElementById('summary-report-container');
    const chartsContainer = document.getElementById('charts-container');
    const uploadPersonaInput = document.getElementById('uploadPersonaFile');
    const downloadPersonaBtn = document.getElementById('downloadPersonaFile');
    const downloadSummaryBtn = document.getElementById('downloadSummaryReport');
    const downloadAllIndividualReportsBtn = document.getElementById('downloadAllIndividualReports');

    // --- 💡 新增：编辑弹窗的选择器 ---
    const modalBackdrop = document.getElementById('editPersonaModalBackdrop');
    const modalPanel = document.getElementById('editPersonaModal');
    const modalForm = document.getElementById('editPersonaForm');
    const modalTitle = document.getElementById('editPersonaTitle');
    const cancelEditBtn = document.getElementById('cancelEditPersona');
    const saveEditBtn = document.getElementById('saveEditPersona');

    // 💡 新增：弹窗表单字段的选择器
    const editFields = {
        age: document.getElementById('edit_age'),
        gender: document.getElementById('edit_gender'),
        city: document.getElementById('edit_city'),
        profession: document.getElementById('edit_profession'),
        education: document.getElementById('edit_education'),
        income: document.getElementById('edit_income'),
        expected_price: document.getElementById('edit_expected_price'),
        drink_frequency: document.getElementById('edit_drink_frequency'),
        drinking_history: document.getElementById('edit_drinking_history'),
        preferred_aroma: document.getElementById('edit_preferred_aroma'),
        mbti: document.getElementById('edit_mbti'),
    };
    
    // --- 💡 新增：用于填充下拉框的固定选项 ---
    const selectOptions = {
        education: ["高中及以下", "大专", "本科", "硕士", "博士"],
        income: ["10万以下", "10-20万", "20-50万", "50万以上"],
        expected_price: ["100元以下", "100-299元", "300-999元", "1000元以上"],
        drink_frequency: ["从不", "每月1-2次", "每周", "几乎每天"],
        preferred_aroma: ["酱香型", "浓香型", "清香型", "其他"],
        gender: ["男", "女", "其他"]
    };

    // --- 3. UI NAVIGATION LOGIC ---
    // ... (goTo 函数保持不变) ...
    const goTo = (step) => {
        steps.forEach(s => s.classList.add('hidden'));
        const targetStepEl = document.getElementById(`step-${step}`);
        if (targetStepEl) {
            targetStepEl.classList.remove('hidden');
        }
        pills.forEach(p => p.classList.toggle('active', p.dataset.step == step));
        window.scrollTo({ top: 0, behavior: 'smooth' });
    };

    // --- 💡 新增：弹窗控制函数 ---
    const showEditModal = () => {
        modalBackdrop.classList.remove('hidden');
        modalPanel.classList.remove('hidden');
    };
    const hideEditModal = () => {
        modalBackdrop.classList.add('hidden');
        modalPanel.classList.add('hidden');
        currentlyEditingPersonaIndex = null; // 重置
    };

    // --- 💡 新增：打开并填充弹窗的函数 ---
    const openEditModalFor = (index) => {
        const persona = personas[index];
        if (!persona) return;
        
        currentlyEditingPersonaIndex = index;
        modalTitle.textContent = `编辑画像 ${index + 1}`;

        // 填充所有字段
        editFields.age.value = persona.age;
        editFields.gender.value = persona.gender;
        editFields.profession.value = persona.profession;
        editFields.education.value = persona.education;
        editFields.income.value = persona.income;
        editFields.expected_price.value = persona.expected_price;
        editFields.drink_frequency.value = persona.drink_frequency;
        editFields.drinking_history.value = persona.drinking_history;
        editFields.preferred_aroma.value = persona.preferred_aroma;
        editFields.mbti.value = persona.mbti;
        
        // 动态填充城市下拉框
        editFields.city.innerHTML = ''; // 清空旧选项
        availableCities.forEach(city => {
            const option = document.createElement('option');
            option.value = city;
            option.textContent = city;
            editFields.city.appendChild(option);
        });
        // 确保所有固定下拉框也有值 (以防万一)
        const populateSelect = (selectEl, options) => {
            selectEl.innerHTML = '';
            options.forEach(opt => {
                const option = document.createElement('option');
                option.value = opt;
                option.textContent = opt;
                selectEl.appendChild(option);
            });
        };
        populateSelect(editFields.gender, selectOptions.gender);
        populateSelect(editFields.education, selectOptions.education);
        populateSelect(editFields.income, selectOptions.income);
        populateSelect(editFields.expected_price, selectOptions.expected_price);
        populateSelect(editFields.drink_frequency, selectOptions.drink_frequency);
        populateSelect(editFields.preferred_aroma, selectOptions.preferred_aroma);

        // 设置选中值
        editFields.city.value = persona.city;
        editFields.gender.value = persona.gender;
        editFields.education.value = persona.education;
        editFields.income.value = persona.income;
        editFields.expected_price.value = persona.expected_price;
        editFields.drink_frequency.value = persona.drink_frequency;
        editFields.preferred_aroma.value = persona.preferred_aroma;

        showEditModal();
    };
    
    // --- 💡 新增：保存画像修改的函数 ---
    const handleSavePersona = () => {
        if (currentlyEditingPersonaIndex === null) return;
        
        // 1. 表单验证
        if (!modalForm.checkValidity()) {
            modalForm.reportValidity(); // 触发浏览器自带的验证提示
            return;
        }

        // 2. 特殊验证
        const age = parseInt(editFields.age.value, 10);
        const history = parseInt(editFields.drinking_history.value, 10);
        const mbti = editFields.mbti.value.toUpperCase();

        if (history > (age - 18)) {
            alert(`验证失败：酒龄 (${history}年) 不能超过 年龄 (${age}岁) 减去 18。`);
            return;
        }
        if (!/^[IE][NS][TF][JP]$/.test(mbti)) {
             alert(`验证失败：MBTI "${editFields.mbti.value}" 不是有效的4字母组合。`);
             return;
        }
        
        // 3. 更新全局 `personas` 数组
        personas[currentlyEditingPersonaIndex] = {
            age: age,
            gender: editFields.gender.value,
            city: editFields.city.value,
            profession: editFields.profession.value,
            education: editFields.education.value,
            income: editFields.income.value,
            expected_price: editFields.expected_price.value,
            drink_frequency: editFields.drink_frequency.value,
            drinking_history: history,
            preferred_aroma: editFields.preferred_aroma.value,
            mbti: mbti,
        };
        
        // 4. 重新渲染概要
        personaSummaryStats = calculateSummaryStats(personas);
        // 如果是通过上传修改的，filePath 会是文件名；如果是生成的，会是 null
        renderPersonaSummary(personaFilePath, personaSummaryStats);
        
        // 5. 关闭弹窗
        hideEditModal();
        alert(`画像 ${currentlyEditingPersonaIndex + 1} 已更新！`);
    };

    // --- 4. PERSONA MANAGEMENT LOGIC (基本不变) ---
    // ... (mbtiRatioLabels, generateButtonLabel, clampPercent, attachRatioInputGuard, applyGuardsToStaticRatios, refreshCityRemoveButtons, updateAddCityButtonState, createCityRow, initializeCitySelector, collectRatioMap, collectCityRatios, buildPersonaConfig, showSummaryPlaceholder, showSummaryError, calculateSummaryStats ... 均保持不变) ...
    
    const mbtiRatioLabels = {
        mbti_energy: 'MBTI 能量倾向', mbti_info: 'MBTI 信息接收',
        mbti_decision: 'MBTI 决策方式', mbti_life: 'MBTI 生活态度',
    };
    const generateButtonLabel = generatePersonasBtn?.innerHTML || '';

    const clampPercent = (value) => {
        const num = Number(value);
        if (Number.isNaN(num)) return 0;
        return Math.max(0, Math.min(100, Math.round(num * 100) / 100));
    };

    const attachRatioInputGuard = (input) => {
        if (!input) return;
        input.addEventListener('input', (event) => {
            const clamped = clampPercent(event.target.value);
            if (event.target.value === '') return;
            if (Number(event.target.value) !== clamped) {
                event.target.value = clamped;
            }
        });
        input.addEventListener('blur', (event) => {
            event.target.value = clampPercent(event.target.value || 0);
        });
    };

    const applyGuardsToStaticRatios = () => {
        document.querySelectorAll('.ratio-input').forEach(attachRatioInputGuard);
    };

    const refreshCityRemoveButtons = () => {
        const rows = cityRatioList ? Array.from(cityRatioList.querySelectorAll('.city-ratio-row')) : [];
        const disable = rows.length <= 1;
        rows.forEach((row) => {
            const removeBtn = row.querySelector('.remove-city');
            if (removeBtn) removeBtn.disabled = disable;
        });
    };

    const updateAddCityButtonState = () => {
        if (!addCityRatioBtn) return;
        const used = new Set(Array.from(cityRatioList.querySelectorAll('select')).map(sel => sel.value));
        const hasRemaining = availableCities.some(city => !used.has(city));
        addCityRatioBtn.disabled = !hasRemaining;
    };

    const createCityRow = (city, ratio) => {
        if (!cityRatioList) return;
        const row = document.createElement('div');
        row.className = 'city-ratio-row';

        const select = document.createElement('select');
        availableCities.forEach(option => { // Use dynamic city list
            const opt = document.createElement('option');
            opt.value = option;
            opt.textContent = option;
            select.appendChild(opt);
        });
        if (city && availableCities.includes(city)) select.value = city;

        const ratioInput = document.createElement('input');
        ratioInput.type = 'number';
        ratioInput.min = '0';
        ratioInput.max = '100';
        ratioInput.value = clampPercent(ratio ?? 0);
        ratioInput.className = 'ratio-input';
        attachRatioInputGuard(ratioInput);

        const removeBtn = document.createElement('button');
        removeBtn.type = 'button';
        removeBtn.className = 'remove-city';
        removeBtn.textContent = '移除';
        removeBtn.addEventListener('click', () => {
            if (cityRatioList.querySelectorAll('.city-ratio-row').length <= 1) return;
            row.remove();
            refreshCityRemoveButtons();
            updateAddCityButtonState();
        });

        select.addEventListener('change', updateAddCityButtonState);

        row.appendChild(select);
        row.appendChild(ratioInput);
        row.appendChild(removeBtn);
        cityRatioList.appendChild(row);
        refreshCityRemoveButtons();
        updateAddCityButtonState();
    };
    
    const initializeCitySelector = async () => {
        try {
            const response = await fetch('/get_city_options');
            if (!response.ok) throw new Error('Failed to fetch cities');
            availableCities = await response.json(); 

            if (!cityRatioList) return;
            cityRatioList.innerHTML = ''; 
            
            const defaultCities = ['北京', '上海', '广州', '深圳'];
            const validDefaults = defaultCities.filter(city => availableCities.includes(city));

            if (validDefaults.length > 0) {
                 const ratio = 100 / validDefaults.length;
                 validDefaults.forEach(city => createCityRow(city, ratio));
            } else if (availableCities.length > 0) {
                const initialCities = availableCities.slice(0, 4);
                const ratio = 100 / initialCities.length;
                initialCities.forEach(city => createCityRow(city, ratio));
            }

        } catch (error) {
            console.error("Could not initialize city selector:", error);
            availableCities = ['北京', '上海', '广州', '深圳', '杭州', '成都', '武汉', '南京', '苏州'];
            const ratio = 25;
            ['北京', '上海', '广州', '深圳'].forEach(city => createCityRow(city, ratio));
        }
    };

    const collectRatioMap = (group) => {
        const inputs = document.querySelectorAll(`.ratio-input[data-group="${group}"]`);
        const map = {};
        inputs.forEach((input) => {
            const optionKey = input.dataset.option;
            const value = clampPercent(input.value || 0);
            input.value = value;
            if (optionKey) map[optionKey] = value;
        });
        return map;
    };

    const collectCityRatios = () => {
        if (!cityRatioList) return [];
        return Array.from(cityRatioList.querySelectorAll('.city-ratio-row')).map((row) => {
            const city = row.querySelector('select')?.value || '';
            const ratioInput = row.querySelector('.ratio-input');
            const ratio = clampPercent(ratioInput?.value || 0);
            if (ratioInput) ratioInput.value = ratio;
            return { city, ratio };
        });
    };

    const buildPersonaConfig = () => {
        const errors = [];
        const count = Number(personaCountEl?.value || 0);
        const ageMin = Number(ageMinEl?.value || 0);
        const ageMax = Number(ageMaxEl?.value || 0);

        if (!Number.isFinite(count) || count < 1) errors.push('画像数量需为正整数。');
        if (!Number.isFinite(ageMin) || !Number.isFinite(ageMax) || ageMin < 18 || ageMax > 80 || ageMin >= ageMax) {
            errors.push('请确保年龄范围在 18-80 岁之间，且最小值小于最大值。');
        }

        const genderRatio = collectRatioMap('gender');
        const drinkRatio = collectRatioMap('drink_frequency');
        const flavorRatio = collectRatioMap('flavor');
        const mbtiRatio = {
            energy: collectRatioMap('mbti_energy'),
            info: collectRatioMap('mbti_info'),
            decision: collectRatioMap('mbti_decision'),
            life: collectRatioMap('mbti_life'),
        };

        const checkSum = (map, label) => {
            const sum = Object.values(map).reduce((acc, value) => acc + value, 0);
            if (Math.abs(sum - 100) > 1.5) {
                errors.push(`${label} 的比例总和需为 100（当前 ${sum.toFixed(2)}）。`);
            }
        };

        checkSum(genderRatio, '性别');
        checkSum(drinkRatio, '饮酒频率');
        checkSum(flavorRatio, '偏好香型');
        Object.entries(mbtiRatio).forEach(([key, map]) => checkSum(map, mbtiRatioLabels[key] || key));
        
        const cityRatios = collectCityRatios();
        if (cityRatios.length === 0) {
            errors.push('请至少配置一个城市比例。');
        } else {
            const citySum = cityRatios.reduce((acc, item) => acc + item.ratio, 0);
            if (Math.abs(citySum - 100) > 1.5) {
                errors.push(`城市比例总和需为 100（当前 ${citySum.toFixed(2)}）。`);
            }
            const seen = new Set();
            const duplicates = new Set();
            cityRatios.forEach(({ city }) => {
                if (seen.has(city)) duplicates.add(city);
                seen.add(city);
            });
            if (duplicates.size > 0) {
                errors.push(`城市 ${Array.from(duplicates).join('、')} 重复，请调整。`);
            }
        }

        return {
            errors,
            config: {
                count: Math.round(count),
                age_range: { min: Math.round(ageMin), max: Math.round(ageMax) },
                gender_ratio: genderRatio,
                mbti_ratio: mbtiRatio,
                drink_frequency_ratio: drinkRatio,
                flavor_ratio: flavorRatio,
                city_ratio: cityRatios.map(entry => ({ city: entry.city, ratio: entry.ratio })),
            }
        };
    };

    const showSummaryPlaceholder = (message) => {
        if (!personaSummaryEl) return;
        personaSummaryEl.classList.add('empty');
        personaSummaryEl.innerHTML = `<div class="placeholder-content"><div class="spinner"></div><p>${message}</p></div>`;
    };

    const showSummaryError = (message) => {
        if (!personaSummaryEl) return;
        personaSummaryEl.classList.remove('empty');
        personaSummaryEl.innerHTML = `<div class="error-message">${message}</div>`;
    };
    
    const calculateSummaryStats = (personasArray) => {
        if (!Array.isArray(personasArray) || personasArray.length === 0) {
            return {};
        }
        const counter = (key) => personasArray.reduce((acc, p) => {
            const value = p[key] || '未知';
            acc[value] = (acc[value] || 0) + 1;
            return acc;
        }, {});
        
        return {
            gender: counter('gender'),
            city: counter('city'),
            drink_frequency: counter('drink_frequency'),
            preferred_aroma: counter('preferred_aroma')
        };
    };

    // --- 💡 更改：`renderPersonaSummary` 函数现在会添加 "编辑" 按钮 ---
    const renderPersonaSummary = (filePath, summary = {}) => {
        if (!personaSummaryEl) return;
        personaSummaryEl.classList.remove('empty');
        personaSummaryEl.innerHTML = '';
        const formatSummary = (obj) => Object.entries(obj).map(([key, value]) => `${key} ${value}人`).join('、');
        const meta = document.createElement('div');
        meta.className = 'summary-meta';
        meta.innerHTML = `已加载 ${personas.length} 个画像。文件：<code>${filePath || '未保存'}</code>`;
        personaSummaryEl.appendChild(meta);
        
        if (Object.keys(summary).length === 0 && personas.length > 0) {
            summary = calculateSummaryStats(personas);
        }

        if (summary && Object.keys(summary).length) {
            const detail = document.createElement('div');
            detail.className = 'summary-meta';
            const segments = [];
            if (summary.gender) segments.push(`性别：${formatSummary(summary.gender)}`);
            if (summary.city) segments.push(`城市：${formatSummary(summary.city)}`);
            if (summary.drink_frequency) segments.push(`饮酒频率：${formatSummary(summary.drink_frequency)}`);
            if (summary.preferred_aroma) segments.push(`香型：${formatSummary(summary.preferred_aroma)}`);
            if (segments.length) {
                detail.innerHTML = segments.join(' &nbsp;|&nbsp; ');
                personaSummaryEl.appendChild(detail);
            }
        }
        const grid = document.createElement('div');
        grid.className = 'summary-grid';
        
        // 💡 更改：循环创建卡片时，添加 "编辑" 按钮
        personas.forEach((persona, index) => {
            const card = document.createElement('div');
            card.className = 'summary-card';
            const items = [
                `${persona.age || '未知'} 岁 · ${persona.gender || '未知性别'}`,
                `城市：${persona.city || '未知'}`, `职业：${persona.profession || '未指定'}`,
                `MBTI：${persona.mbti || '未指定'}`, `饮酒频率：${persona.drink_frequency || '未指定'}`,
                `偏好香型：${persona.preferred_aroma || '未指定'}`
            ];
            
            // 💡 新增：编辑按钮的 HTML
            const editButtonHtml = `
                <button class="btn-edit-persona" data-persona-index="${index}" title="编辑画像 ${index + 1}">
                    <svg class="icon"><use href="#i-edit"/></svg>
                </button>
            `;
            
            card.innerHTML = `
                ${editButtonHtml}
                <h4>画像 ${index + 1}</h4>
                <ul>${items.map(item => `<li>${item}</li>`).join('')}</ul>
            `;
            grid.appendChild(card);
        });
        personaSummaryEl.appendChild(grid);
    };

    // ... (handleGeneratePersonas 函数保持不变) ...
    const handleGeneratePersonas = async () => {
        const { errors, config } = buildPersonaConfig();
        if (errors.length > 0) {
            alert(errors.join('\n'));
            return;
        }
        try {
            if (generatePersonasBtn) {
                generatePersonasBtn.disabled = true;
                generatePersonasBtn.textContent = '生成中...';
            }
            showSummaryPlaceholder('正在根据比例生成画像，请稍候...');
            const response = await fetch('/generate_personas', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(config),
            });
            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(JSON.parse(errorText).error || `接口返回状态 ${response.status}`);
            }
            const result = await response.json();
            personas = Array.isArray(result.personas) ? result.personas : [];
            personaFilePath = result.file || null; 
            personaSummaryStats = result.summary || {};
            if (!personas.length) throw new Error('未生成任何画像，请调整参数后重试。');
            renderPersonaSummary(personaFilePath, personaSummaryStats);
        } catch (error) {
            console.error('生成画像失败:', error);
            personas = [];
            showSummaryError(`生成画像失败：${error.message || error}`);
        } finally {
            if (generatePersonasBtn) {
                generatePersonasBtn.disabled = false;
                generatePersonasBtn.innerHTML = generateButtonLabel;
            }
        }
    };
    

    // --- 5. PRODUCT INFO LOGIC ---
    // ... (handleImageUpload 函数保持不变) ...
    const handleImageUpload = (event) => {
        const file = event.target.files[0];
        if (!file) {
            productPreviewEl.innerHTML = '建议上传横向高清图，展示瓶身与包装细节。';
            productPreviewEl.classList.add('tip');
            productImageDataURL = null;
            return;
        }
        const reader = new FileReader();
        reader.onload = () => {
            productImageDataURL = reader.result;
            productPreviewEl.innerHTML = `<img src="${productImageDataURL}" alt="产品图片预览">`;
            productPreviewEl.classList.remove('tip');
        };
        reader.readAsDataURL(file);
    };
    productImgInput?.addEventListener('change', handleImageUpload);


    // --- 6. REPORT GENERATION & STREAMING LOGIC ---
    // ... (generateReport 函数保持不变) ...
    const generateReport = async () => {
        if (personas.length === 0) { alert('请先生成或上传画像后再继续。'); return; }
        if (!productDescEl.value.trim() || !productImageDataURL) { alert('请完善产品描述并上传产品图片。'); return; }
        
        goTo('3');
        
        individualReportsRendered = 0;
        individualContainer.innerHTML = '<div class="placeholder-content" id="initial-loader"><div class="spinner"></div><p>正在启动分析引擎，请稍候...</p></div>';
        summaryContainer.innerHTML = '<div class="placeholder-content"><div class="spinner"></div><p>等待独立报告生成完毕...</p></div>';
        chartsContainer.innerHTML = '';
        nextToSummaryContainer.style.display = 'none';

        const payload = { 
            personas: personas, 
            persona_file: null, 
            productData: { 
                description: productDescEl.value, 
                image: productImageDataURL 
            } 
        };

        try {
            const startResponse = await fetch('/generate_report', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
            if (!startResponse.ok) throw new Error(`启动分析任务失败: ${startResponse.statusText}`);
            const { job_id } = await startResponse.json();
            const eventSource = new EventSource(`/stream/${job_id}`);

            eventSource.onmessage = (event) => {
                const message = JSON.parse(event.data);
                
                switch (message.type) {
                    case 'individual_report':
                        if (individualReportsRendered === 0) individualContainer.innerHTML = '';
                        displayIndividualReport(message.data);
                        individualReportsRendered++;
                        if (individualReportsRendered < personas.length) {
                             const existingLoader = document.getElementById('next-report-loader');
                             if (!existingLoader) {
                                individualContainer.insertAdjacentHTML('beforeend', `<div id="next-report-loader" class="persona-report-loading"><div class="spinner-small"></div><span>正在生成下一个画像分析报告...</span></div>`);
                             }
                        } else {
                            document.getElementById('next-report-loader')?.remove();
                            nextToSummaryContainer.style.display = 'flex';
                        }
                        break;
                    case 'summary_report': 
                        goTo('4');
                        displaySummaryReport(message.data); 
                        break;
                    case 'chart_and_table': 
                        displayChartAndTable(message.data); 
                        break;
                    case 'table_analysis': 
                        displayTableAnalysis(message.data); 
                        break;
                    case 'done': 
                        eventSource.close();
                        document.getElementById('next-report-loader')?.remove();
                        if (!document.querySelector('#summary-report-container .prose p')) {
                             summaryContainer.innerHTML = '<p>分析已完成。</p>';
                        }
                        break;
                    case 'error':
                        individualContainer.innerHTML = `<div class="error-message">分析出错: ${message.data}</div>`;
                        summaryContainer.innerHTML = `<div class="error-message">分析出错: ${message.data}</div>`;
                        eventSource.close();
                        break;
                }
            };
            eventSource.onerror = (err) => {
                console.error("数据流连接错误:", err);
                individualContainer.innerHTML = `<div class="error-message">与服务器的连接丢失，请刷新页面重试。</div>`;
                eventSource.close();
            };
        } catch (error) {
            console.error("无法启动分析任务:", error);
            individualContainer.innerHTML = `<div class="error-message">无法启动分析任务，请检查后端服务是否正在运行。</div>`;
        }
    };

    // --- 7. DISPLAY & RENDER LOGIC (下载功能不变) ---
    // ... (downloadAsHTML, displayIndividualReport, displaySummaryReport, displayChartAndTable, displayTableAnalysis ... 均保持不变) ...
    const downloadAsHTML = (content, filename, title) => {
        const css = `
            <style>
                body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; line-height: 1.6; padding: 20px; max-width: 800px; margin: auto; color: #333; }
                h1, h2, h3, h4, h5 { color: #6E0F1A; font-family: "Noto Serif SC", serif; }
                h1 { font-size: 24px; } h5 { font-size: 16px; margin-bottom: 5px; }
                img { max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
                p { margin-top: 0; }
                hr { border: 0; border-top: 1px dashed #ccc; margin: 20px 0; }
                .report-header-actions { border-bottom: 1px solid #eee; padding-bottom: 10px; }
                .persona-report { border-bottom: 2px solid #6E0F1A; margin-bottom: 20px; padding-bottom: 20px; }
                .report-details-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; align-items: center; margin: 16px 0; }
                .final-decision-section { border-top: 1px dashed #ccc; padding-top: 10px; margin-top: 10px; }
                .decision-buy { color: green; font-weight: bold; }
                .decision-nobuy { color: red; font-weight: bold; }
                .prose p { margin-bottom: 1em; }
                .charts-section img { max-width: 450px; display: block; margin: 10px auto; }
                .kv { border: 1px solid #ddd; border-radius: 8px; overflow: hidden; margin: 10px 0; }
                .kv .row { display: grid; grid-template-columns: 1fr 90px; border-bottom: 1px solid #eee; }
                .kv .row:last-child { border-bottom: none; }
                .kv .row div { padding: 8px 10px; }
                .note { background: #fef9e7; border: 1px solid #f7dc6f; border-radius: 4px; padding: 10px; font-size: 14px; }
                @media (max-width: 600px) {
                    .report-details-grid { grid-template-columns: 1fr; }
                }
            </style>
        `;
        const html = `
            <!DOCTYPE html>
            <html lang="zh-CN">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>${title}</title>
                ${css}
            </head>
            <body>
                <h1>${title}</h1>
                ${content}
            </body>
            </html>
        `;
        const blob = new Blob([html], { type: 'text/html' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    };

    const displayIndividualReport = (data) => {
        const reportEl = document.createElement('div');
        reportEl.className = 'persona-report';
        reportEl.id = `persona-report-${data.persona_id}`; 

        const persona = data.persona_details;
        const decisionClass = data.decision === '购买' ? 'decision-buy' : 'decision-nobuy';
        const reportContent = data.report || {};
        
        let introParts = [];
        if (persona.age) introParts.push(`${persona.age}岁`);
        if (persona.gender) introParts.push(persona.gender);
        if (persona.city) introParts.push(`来自${persona.city}`);
        if (persona.profession) introParts.push(`职业: ${persona.profession}`);
        if (persona.education) introParts.push(`教育: ${persona.education}`);
        if (persona.income) introParts.push(`年收入: ${persona.income}`);
        if (persona.expected_price) introParts.push(`心理价位: ${persona.expected_price}`);
        if (persona.mbti) introParts.push(`MBTI: ${persona.mbti}`);
        if (persona.drink_frequency) introParts.push(`饮酒频率: ${persona.drink_frequency}`);
        if (persona.drinking_history !== undefined && persona.drinking_history !== null) {
             introParts.push(`酒龄: ${persona.drinking_history}年`);
        }
        if (persona.preferred_aroma) introParts.push(`偏好香型: ${persona.preferred_aroma}`);
        let personaIntro = introParts.length > 0 ? introParts.join('，') + '。' : '画像信息不完整。';

        reportEl.innerHTML = `
            <div class="report-header-actions">
                <h4 class="h4">画像 ${data.persona_id} 独立报告</h4>
                </div>
            <p class="muted">画像简介：${personaIntro}</p>
            <div class="report-content-wrapper"> 
                <div class="report-details-grid">
                    <div class="report-text-sections">
                        <div class="report-section"><h5>包装视觉评估</h5><p>${reportContent.packaging_analysis || "AI未提供此项分析。"}</p></div>
                        <div class="report-section"><h5>产品契合度分析</h5><p>${reportContent.fit_analysis || "AI未提供此项分析。"}</p></div>
                        <div class="report-section"><h5>潜在消费场景</h5><p>${reportContent.scenario_analysis || "AI未提供此项分析。"}</p></div>
                    </div>
                    ${data.radar_chart ? `<div class="radar-chart-container"><h5>画像-产品匹配度雷达图</h5> <img src="data:image/png;base64,${data.radar_chart}" alt="雷达图"></div>` : ''}
                </div>
                <div class="final-decision-section">
                    <p><strong class="${decisionClass}">【最终决策】</strong> ${data.final_decision?.decision || '未明确'}</p>
                    <p><strong>【决策理由】</strong> ${data.final_decision?.reason || 'AI未提供理由。'}</p>
                </div>
            </div>`;
        
        individualContainer.appendChild(reportEl);
    };
    
    const displaySummaryReport = (data) => {
        summaryContainer.classList.remove('placeholder-content');
        summaryContainer.innerHTML = `<div class="prose"><p>${data.replace(/\n\n/g, '</p><p>').replace(/\n/g, '<br>')}</p></div>`;
    };
    
    const displayChartAndTable = (data) => {
        let container = document.getElementById(`chart-container-${data.id}`);
        if (!container) {
            container = document.createElement('div');
            container.id = `chart-container-${data.id}`;
            chartsContainer.appendChild(container);
        }
        const tableRows = Object.entries(data.table).map(([key, value]) => `<div class="row"><div>${key}</div><div>${value}</div></div>`).join('');
        container.innerHTML = `
            <h4 class="h4">${data.title}</h4>
            ${data.chart ? `<img src="data:image/png;base64,${data.chart}" alt="${data.title}" style="max-width:100%; border-radius:12px; margin-bottom:12px; border:1px solid var(--line);">` : ''}
            <div class="kv">${tableRows}</div>
            <div class="note placeholder-note" id="analysis-placeholder-${data.id}">
                <div class="spinner-small" style="width:12px; height:12px; border-width:2px; margin-right:6px;"></div> 正在生成AI洞察...
            </div>`;
    };

    const displayTableAnalysis = (data) => {
        const placeholder = document.getElementById(`analysis-placeholder-${data.id}`);
        if (placeholder) {
            placeholder.innerHTML = `<strong>AI洞察：</strong> ${data.analysis}`;
            placeholder.classList.remove('placeholder-note');
        }
    };
    
    // --- 8. EVENT LISTENERS ---
    // ... (导航按钮, 添加城市, 生成画像, 生成报告 ... 监听器保持不变) ...
    document.querySelectorAll('.next, .prev, .step-pill').forEach(btn => {
        btn.addEventListener('click', e => {
            const targetStep = e.currentTarget.dataset.next || e.currentTarget.dataset.prev || e.currentTarget.dataset.step;
            if (!targetStep) return;
            if (e.currentTarget.dataset.next === '2' && personas.length === 0) {
                alert('请先生成或上传画像，再进入下一步。');
                return;
            }
            goTo(targetStep);
        });
    });
    addCityRatioBtn?.addEventListener('click', () => {
        const used = new Set(Array.from(cityRatioList.querySelectorAll('select')).map(sel => sel.value));
        const nextCity = availableCities.find(city => !used.has(city));
        if (nextCity) {
            createCityRow(nextCity, 0);
        }
    });
    generatePersonasBtn?.addEventListener('click', handleGeneratePersonas);
    genReportBtn?.addEventListener('click', generateReport);


    // ... (画像上传/下载, 报告下载 ... 监听器保持不变) ...
    downloadPersonaBtn?.addEventListener('click', () => {
        if (personas.length === 0) {
            alert('当前没有画像可供下载。');
            return;
        }
        const content = JSON.stringify(personas, null, 2);
        const blob = new Blob([content], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `personas_${new Date().strftime('%Y%m%d_%H%M')}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    });

    uploadPersonaInput?.addEventListener('change', (event) => {
        const file = event.target.files[0];
        if (!file) return;
        const reader = new FileReader();
        reader.onload = (e) => {
            try {
                const parsedPersonas = JSON.parse(e.target.result);
                if (!Array.isArray(parsedPersonas) || parsedPersonas.length === 0) {
                    throw new Error('文件内容不是有效的画像数组或数组为空。');
                }
                personas = parsedPersonas; 
                personaFilePath = file.name; 
                personaSummaryStats = calculateSummaryStats(personas); 
                renderPersonaSummary(personaFilePath, personaSummaryStats);
                alert(`成功加载 ${personas.length} 个画像！`);
            } catch (error) {
                console.error('上传画像失败:', error);
                alert(`无法读取文件：${error.message}`);
                personas = []; 
                showSummaryError(`文件加载失败：${error.message}`);
            } finally {
                event.target.value = null;
            }
        };
        reader.readAsText(file);
    });

    downloadAllIndividualReportsBtn?.addEventListener('click', () => {
        if (individualReportsRendered === 0 || !individualContainer) {
            alert('没有可下载的独立报告。');
            return;
        }
        const contentClone = individualContainer.cloneNode(true);
        const loader = contentClone.querySelector('#next-report-loader');
        if (loader) {
            loader.remove();
        }
        const content = contentClone.innerHTML;
        if (!content.trim()) {
             alert('没有可下载的报告内容。');
             return;
        }
        downloadAsHTML(content, 'all_persona_reports.html', '全体画像独立分析报告');
    });

    downloadSummaryBtn?.addEventListener('click', () => {
        const summaryContent = summaryContainer.querySelector('.prose')?.innerHTML;
        const chartsContent = chartsContainer.innerHTML; 
        if ((!summaryContent || summaryContent.includes('placeholder-content')) && !chartsContent) {
            alert('没有可下载的报告或图表。');
            return;
        }
        const combinedContent = `
            <div class="report-section">
                ${summaryContent || '<p>未生成报告文本。</p>'}
            </div>
            <hr>
            <div class="charts-section">
                <h2>数据洞察可视化</h2>
                ${chartsContent || '<p>未生成图表。</p>'}
            </div>
        `;
        downloadAsHTML(combinedContent, 'summary_report_with_charts.html', '综合市场分析与数据可视化报告');
    });
    
    // --- 💡 新增：弹窗和编辑按钮的事件监听器 ---
    
    // 1. 监听保存按钮
    modalForm.addEventListener('submit', (e) => {
        e.preventDefault(); // 阻止表单默认提交
        handleSavePersona();
    });
    
    // 2. 监听取消按钮
    cancelEditBtn.addEventListener('click', hideEditModal);
    
    // 3. 监听弹窗背景
    modalBackdrop.addEventListener('click', hideEditModal);

    // 4. 监听卡片上的编辑按钮 (使用事件委托)
    personaSummaryEl.addEventListener('click', (e) => {
        const editButton = e.target.closest('.btn-edit-persona');
        if (editButton) {
            const index = parseInt(editButton.dataset.personaIndex, 10);
            if (!isNaN(index)) {
                openEditModalFor(index);
            }
        }
    });

    // --- INITIALIZATION ---
    applyGuardsToStaticRatios();
    initializeCitySelector(); // Fetch cities and setup the UI

    // --- Helper for formatting date in download ---
    Date.prototype.strftime = function(format) {
        const d = this;
        const pad = (num) => (num < 10 ? '0' : '') + num;
        return format.replace(/%Y|%m|%d|%H|%M/g, (match) => {
            switch (match) {
                case '%Y': return d.getFullYear();
                case '%m': return pad(d.getMonth() + 1);
                case '%d': return pad(d.getDate());
                case '%H': return pad(d.getHours());
                case '%M': return pad(d.getMinutes());
                default: return match;
            }
        });
    };
});