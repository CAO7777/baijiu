// static/script.js (v4.4 - Real Data Integration)

document.addEventListener('DOMContentLoaded', () => {
    // --- 1. STATE MANAGEMENT ---
    let personas = [];
    let personaFilePath = null;
    let personaSummaryStats = {};
    let individualReportsRendered = 0;
    let productImageDataURL = null;
    let availableCities = []; // NEW: To store cities from the server

    // --- 2. DOM ELEMENT SELECTORS ---
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
    const exportBtn = document.getElementById('exportBtn');

    // --- 3. UI NAVIGATION LOGIC ---
    const goTo = (step) => {
        steps.forEach(s => s.classList.add('hidden'));
        const targetStepEl = document.getElementById(`step-${step}`);
        if (targetStepEl) {
            targetStepEl.classList.remove('hidden');
        }
        pills.forEach(p => p.classList.toggle('active', p.dataset.step == step));
        window.scrollTo({ top: 0, behavior: 'smooth' });
    };

    // --- 4. PERSONA MANAGEMENT LOGIC (MODIFIED FOR DYNAMIC CITIES) ---
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
    
    // NEW: Function to fetch cities and initialize the UI
    const initializeCitySelector = async () => {
        try {
            const response = await fetch('/get_city_options');
            if (!response.ok) throw new Error('Failed to fetch cities');
            availableCities = await response.json();

            if (!cityRatioList) return;
            cityRatioList.innerHTML = ''; // Clear any existing rows
            
            // Set up default cities if they exist in the new dynamic list
            const defaultCities = ['北京', '上海', '广州', '深圳'];
            const validDefaults = defaultCities.filter(city => availableCities.includes(city));
            
            if (validDefaults.length > 0) {
                 const ratio = 100 / validDefaults.length;
                 validDefaults.forEach(city => createCityRow(city, ratio));
            } else if (availableCities.length > 0) {
                // If no defaults are valid, just add the first few cities
                const initialCities = availableCities.slice(0, 4);
                const ratio = 100 / initialCities.length;
                initialCities.forEach(city => createCityRow(city, ratio));
            }

        } catch (error) {
            console.error("Could not initialize city selector:", error);
            // Fallback to a hardcoded list in case of error
            availableCities = ['北京', '上海', '广州', '深圳', '杭州', '成都'];
            const ratio = 100 / 4;
            availableCities.slice(0, 4).forEach(city => createCityRow(city, ratio));
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
        // This function works as is, no changes needed.
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

    const renderPersonaSummary = (filePath, summary = {}) => {
        if (!personaSummaryEl) return;
        personaSummaryEl.classList.remove('empty');
        personaSummaryEl.innerHTML = '';
        const formatSummary = (obj) => Object.entries(obj).map(([key, value]) => `${key} ${value}人`).join('、');
        const meta = document.createElement('div');
        meta.className = 'summary-meta';
        meta.innerHTML = `已生成 ${personas.length} 个画像。JSON 文件：<code>${filePath || '未保存'}</code>`;
        personaSummaryEl.appendChild(meta);
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
        personas.forEach((persona, index) => {
            const card = document.createElement('div');
            card.className = 'summary-card';
            const items = [
                `${persona.age || '未知'} 岁 · ${persona.gender || '未知性别'}`,
                `城市：${persona.city || '未知'}`, `职业：${persona.profession || '未指定'}`,
                `MBTI：${persona.mbti || '未指定'}`, `饮酒频率：${persona.drink_frequency || '未指定'}`,
                `偏好香型：${persona.preferred_aroma || '未指定'}`
            ];
            card.innerHTML = `<h4>画像 ${index + 1}</h4><ul>${items.map(item => `<li>${item}</li>`).join('')}</ul>`;
            grid.appendChild(card);
        });
        personaSummaryEl.appendChild(grid);
    };

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
    const generateReport = async () => {
        if (personas.length === 0) { alert('请先生成画像后再继续。'); return; }
        if (!productDescEl.value.trim() || !productImageDataURL) { alert('请完善产品描述并上传产品图片。'); return; }
        
        goTo('3');
        
        individualReportsRendered = 0;
        individualContainer.innerHTML = '<div class="placeholder-content" id="initial-loader"><div class="spinner"></div><p>正在启动分析引擎，请稍候...</p></div>';
        summaryContainer.innerHTML = '<div class="placeholder-content"><div class="spinner"></div><p>等待独立报告生成完毕...</p></div>';
        chartsContainer.innerHTML = '';
        nextToSummaryContainer.style.display = 'none';

        const payload = { personas, persona_file: personaFilePath, productData: { description: productDescEl.value, image: productImageDataURL } };

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
                        goTo('4'); // Automatically go to summary page
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
                             summaryContainer.innerHTML = '<p>分析已完成。</p>'; // Handle case where summary might not have been generated
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

    // --- 7. DISPLAY & RENDER LOGIC ---
const displayIndividualReport = (data) => {
        const reportEl = document.createElement('div');
        reportEl.className = 'persona-report';

        const persona = data.persona_details; // 获取完整的画像信息
        const decisionClass = data.decision === '购买' ? 'decision-buy' : 'decision-nobuy';
        const reportContent = data.report || {};

        // --- *** 修改：构建更全面的用户简介，显示所有字段 *** ---
        let introParts = []; // 使用数组来构建简介，方便处理顺序和分隔符

        // 基本信息
        if (persona.age) introParts.push(`${persona.age}岁`);
        if (persona.gender) introParts.push(persona.gender);
        if (persona.city) introParts.push(`来自${persona.city}`);

        // 职业与教育
        if (persona.profession) introParts.push(`职业: ${persona.profession}`);
        if (persona.education) introParts.push(`教育: ${persona.education}`);

        // 经济状况
        if (persona.income) introParts.push(`年收入: ${persona.income}`);
        if (persona.expected_price) introParts.push(`心理价位: ${persona.expected_price}`);

        // 性格
        if (persona.mbti) introParts.push(`MBTI: ${persona.mbti}`);

        // 饮酒习惯
        if (persona.drink_frequency) introParts.push(`饮酒频率: ${persona.drink_frequency}`);
        // 检查 drinking_history 是否存在且不为 null/undefined (0是有效值)
        if (persona.drinking_history !== undefined && persona.drinking_history !== null) {
             introParts.push(`酒龄: ${persona.drinking_history}年`);
        }
        if (persona.preferred_aroma) introParts.push(`偏好香型: ${persona.preferred_aroma}`);

        // 将所有部分用逗号连接，并在末尾加上句号
        let personaIntro = introParts.length > 0 ? introParts.join('，') + '。' : '画像信息不完整。';
        // --- *** 用户简介修改结束 *** ---

        reportEl.innerHTML = `
            <h4 class="h4">画像 ${data.persona_id} 独立报告</h4>
            {/* --- *** 使用新的、更全面的简介 *** --- */}
            <p class="muted">画像简介：${personaIntro}</p>
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
    document.querySelectorAll('.next, .prev, .step-pill').forEach(btn => {
        btn.addEventListener('click', e => {
            const targetStep = e.currentTarget.dataset.next || e.currentTarget.dataset.prev || e.currentTarget.dataset.step;
            if (!targetStep) return;
            if (e.currentTarget.dataset.next === '2' && personas.length === 0) {
                alert('请先生成符合要求的画像，再进入下一步。');
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
    exportBtn?.addEventListener('click', () => alert('导出功能待实现。'));
    
    // --- INITIALIZATION ---
    applyGuardsToStaticRatios();
    initializeCitySelector(); // Fetch cities and setup the UI
});