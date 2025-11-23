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

    let isAdjusting = false;

    // --- 2. DOM ELEMENT SELECTORS ---

    const brandSelect = document.getElementById('brandSelect');
    const brandPolyu = document.getElementById('brand-polyu');
    const brandMingyin = document.getElementById('brand-mingyin');

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

    const cotModal = document.getElementById('cotModal');
    const cotContent = document.getElementById('cotContent');
    const cotTextSpan = cotContent?.querySelector('.cot-text');
    let cotEventSource = null; // 用于存储 CoT 的 SSE 连接

    // --- 💡 新增：CoT 控制函数 ---
    const startCotStream = (jobId) => {
        if (!cotModal || !cotTextSpan) return;
        
        cotTextSpan.innerHTML = ''; // 清空旧内容
        cotModal.classList.remove('hidden'); // 显示浮窗
        
        // 连接到 Flask 的 Fake CoT 接口
        cotEventSource = new EventSource(`/stream_cot/${jobId}`);
        
        cotEventSource.onmessage = (event) => {
            // 追加文本，模拟打字机效果
            // 后端发来的 <br/> 可能会被转义，这里简单 append 即可，因为我们用了 innerHTML
            // 但要注意安全性，这里是受控的 AI 输出，暂时忽略 XSS
            cotTextSpan.innerHTML += event.data; 
            
            // 自动滚动到底部
            if (cotContent) {
                cotContent.scrollTop = cotContent.scrollHeight;
            }
        };

        cotEventSource.onerror = (err) => {
            console.log("CoT Stream error (maybe closed by server):", err);
            // 如果出错，不要关闭浮窗，让它停留在最后的状态，等待真实报告将其关闭
            if (cotEventSource) cotEventSource.close();
        };
    };

    const stopCotStream = () => {
        if (cotEventSource) {
            cotEventSource.close();
            cotEventSource = null;
        }
        if (cotModal) {
            cotModal.classList.add('hidden'); // 隐藏浮窗
        }
    };

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


const formatNumber = (num, decimals = 1) => parseFloat(num.toFixed(decimals));

    // 核心函数：调整一组输入/滑块以保持总和为 100
    const adjustGroupRatios = (changedInput) => {
        if (isAdjusting) return; // 防止无限循环
        isAdjusting = true;

        const groupContainer = changedInput.closest('.ratio-with-slider, .city-ratio-list');
        if (!groupContainer) {
            isAdjusting = false;
            return;
        }

        const isCityGroup = groupContainer.id === 'cityRatioList';
        const groupSelector = isCityGroup ? '.city-ratio-row' : '.field-slider-group';
        const allGroups = Array.from(groupContainer.querySelectorAll(groupSelector));
        const inputs = allGroups.map(g => g.querySelector('.ratio-input')).filter(Boolean);
        const sliders = allGroups.map(g => g.querySelector('.ratio-slider')).filter(Boolean);

        const currentIndex = inputs.indexOf(changedInput.type === 'number' ? changedInput : inputs.find(inp => inp.closest(groupSelector) === changedInput.closest(groupSelector)));
        if (currentIndex === -1) {
             isAdjusting = false;
             return; // Should not happen
        }
        
        const currentValue = clampPercent(parseFloat(changedInput.value || 0));
        inputs[currentIndex].value = currentValue; // 确保数字框的值被修正
        if (sliders[currentIndex]) sliders[currentIndex].value = currentValue; // 同步滑块

        const otherInputs = inputs.filter((_, idx) => idx !== currentIndex);
        const otherSliders = sliders.filter((_, idx) => idx !== currentIndex);

        let currentTotal = inputs.reduce((sum, inp) => sum + parseFloat(inp.value || 0), 0);
        let delta = 100 - currentTotal; // 需要调整的总量

        if (Math.abs(delta) < 0.01) { // 已经在 100 附近，无需调整
             isAdjusting = false;
             return;
        }

        // --- 分配逻辑 ---
        if (otherInputs.length === 1) { // 只有两项的情况 (性别, MBTI)
            const otherValue = clampPercent(parseFloat(otherInputs[0].value || 0) + delta);
            otherInputs[0].value = otherValue;
            if (otherSliders[0]) otherSliders[0].value = otherValue;
        } else if (otherInputs.length > 1) { // 多项的情况 (频率, 香型, 城市)
            let otherTotal = otherInputs.reduce((sum, inp) => sum + parseFloat(inp.value || 0), 0);
            
            // 如果所有其他项都是0，则无法按比例分配，平均分配（或不处理，让总和暂时不为100）
            // 这里我们选择让总和暂时不为100，让用户继续调整
            if (otherTotal <= 0 && delta < 0) { // 无法从0中扣除
                 console.warn("Cannot decrease other values as they sum to zero.");
            } else {
                 let remainingDelta = delta;
                 let adjustedValues = [];

                 // 按比例分配 delta
                 for (let i = 0; i < otherInputs.length; i++) {
                     const input = otherInputs[i];
                     const slider = otherSliders[i];
                     const oldValue = parseFloat(input.value || 0);
                     let share = 0;
                     if (delta > 0) { // 需要增加，按 (100 - oldValue) 比例? 或平均? 简单起见先平均分
                         share = delta / otherInputs.length;
                     } else if (otherTotal > 0) { // 需要减少，且有值可减
                         share = (oldValue / otherTotal) * delta; // 按当前比例减
                     }
                     
                     let newValue = clampPercent(oldValue + share);
                     // 避免因为浮点数精度导致越界
                     if (delta < 0 && newValue < 0) newValue = 0;
                     if (delta > 0 && newValue > 100) newValue = 100;
                     
                     adjustedValues.push(newValue);
                 }

                 // 由于浮点数精度，重新计算总和并调整最后一项
                 let adjustedSum = currentValue + adjustedValues.reduce((sum, val) => sum + val, 0);
                 let finalDelta = 100 - adjustedSum;

                 if (Math.abs(finalDelta) > 0.01 && adjustedValues.length > 0) {
                     let lastIndex = adjustedValues.length - 1;
                     adjustedValues[lastIndex] = clampPercent(adjustedValues[lastIndex] + finalDelta);
                     // 再次确保不越界
                     if (adjustedValues[lastIndex] < 0) adjustedValues[lastIndex] = 0;
                     if (adjustedValues[lastIndex] > 100) adjustedValues[lastIndex] = 100;
                 }

                 // 应用调整后的值
                 for (let i = 0; i < otherInputs.length; i++) {
                      const formattedVal = formatNumber(adjustedValues[i], 1); // 保留一位小数
                      otherInputs[i].value = formattedVal;
                      if (otherSliders[i]) otherSliders[i].value = formattedVal;
                 }
            }
        }
        
        // 最终检查并强制总和为 100 (处理极端情况或精度问题)
        let finalSum = inputs.reduce((sum, inp) => sum + parseFloat(inp.value || 0), 0);
        let finalCorrection = 100 - finalSum;
        if (Math.abs(finalCorrection) > 0.01 && inputs.length > 0) {
             let lastInput = inputs[inputs.length - 1];
             let lastSlider = sliders[sliders.length - 1];
             let correctedLastValue = clampPercent(parseFloat(lastInput.value || 0) + finalCorrection);
             lastInput.value = formatNumber(correctedLastValue, 1);
             if (lastSlider) lastSlider.value = formatNumber(correctedLastValue, 1);
        }

        isAdjusting = false;
    };

    // 事件处理函数：同步输入框和滑块，并触发联动调整
const handleRatioInputChange = (event) => {
        const input = event.target;
        const group = input.closest('.field-slider-group, .city-ratio-row');
        if (!group || isAdjusting) return; // Prevent loops if called during adjustment

        const slider = group.querySelector('.ratio-slider');
        const numberInput = group.querySelector('.ratio-input');
        const groupContainer = input.closest('.ratio-with-slider, .city-ratio-list');

        if (input.type === 'range') { // Change came from SLIDER
            const sliderValue = clampPercent(parseFloat(input.value || 0));
            input.value = sliderValue; // Ensure slider value is clamped
            if (numberInput) {
                numberInput.value = sliderValue; // Sync number input
            }
            // 💡 ONLY sliders trigger automatic adjustment of others
            adjustGroupRatiosFromSlider(input); 
        } else if (input.type === 'number') { // Change came from NUMBER INPUT
             const numberValue = clampPercent(parseFloat(input.value || 0));
             // Don't immediately reformat input.value here, allow user to finish typing
             if (event.type === 'change' || event.type === 'blur') { // On completion, format and clamp
                  input.value = numberValue;
             }
             if (slider) {
                 slider.value = numberValue; // Sync slider
             }
             // 💡 Number inputs DO NOT trigger auto-adjustment, just update total display
             updateTotalDisplay(groupContainer); 
        }
    };

    const handleNumberInputBlur = (event) => {
         const input = event.target;
         if (input.type === 'number' && !isAdjusting) {
             const finalValue = clampPercent(parseFloat(input.value || 0));
             input.value = finalValue; // Set the final clamped value
             const group = input.closest('.field-slider-group, .city-ratio-row');
             if(group) {
                  const slider = group.querySelector('.ratio-slider');
                  if (slider) slider.value = finalValue; // Ensure slider matches
                  const groupContainer = input.closest('.ratio-with-slider, .city-ratio-list');
                  updateTotalDisplay(groupContainer); // Update total display based on final value
             }
         }
     };


    // --- 4. PERSONA MANAGEMENT LOGIC (基本不变) ---
    // ... (mbtiRatioLabels, generateButtonLabel, clampPercent, attachRatioInputGuard, applyGuardsToStaticRatios, refreshCityRemoveButtons, updateAddCityButtonState, createCityRow, initializeCitySelector, collectRatioMap, collectCityRatios, buildPersonaConfig, showSummaryPlaceholder, showSummaryError, calculateSummaryStats ... 均保持不变) ...
    
    const mbtiRatioLabels = {
        mbti_energy: 'MBTI 能量倾向', mbti_info: 'MBTI 信息接收',
        mbti_decision: 'MBTI 决策方式', mbti_life: 'MBTI 生活态度',
    };
    const generateButtonLabel = generatePersonasBtn?.innerHTML || '';

const clampPercent = (value) => {
        const num = parseFloat(value); // Use parseFloat
        if (isNaN(num)) return 0;
        return Math.max(0, Math.min(100, formatNumber(num, 1))); // Clamp and format
    };

const updateTotalDisplay = (groupContainer) => {
        if (!groupContainer) return;
        
        const isCityGroup = groupContainer.id === 'cityRatioList';
        const groupSelector = isCityGroup ? '.city-ratio-row' : '.field-slider-group';
        const allGroups = Array.from(groupContainer.querySelectorAll(groupSelector));
        const inputs = allGroups.map(g => g.querySelector('.ratio-input')).filter(Boolean);
        
        const currentTotal = inputs.reduce((sum, inp) => sum + parseFloat(inp.value || 0), 0);
        const totalFormatted = formatNumber(currentTotal, 1);

        let displayEl = null;
        if (isCityGroup) {
            // Find the hint paragraph in the city group header
             displayEl = groupContainer.previousElementSibling?.querySelector('.ratio-total-display');
             if (!displayEl) { // If not found, try finding the hint directly if structure changed
                 displayEl = groupContainer.closest('.group')?.querySelector('.group-head .hint .ratio-total-display');
             }
        } else {
            // Find the hint span in the ratio group header
            displayEl = groupContainer.closest('.ratio-group')?.querySelector('.ratio-head .hint .ratio-total-display');
        }

        if (displayEl) {
            displayEl.textContent = `当前总计: ${totalFormatted}`;
            // Add visual feedback based on the total
            const parentHint = displayEl.parentElement; // The hint span/p
             if (Math.abs(totalFormatted - 100) < 0.01) {
                 parentHint.classList.remove('invalid-total');
                 parentHint.classList.add('valid-total');
             } else {
                 parentHint.classList.remove('valid-total');
                 parentHint.classList.add('invalid-total');
             }
        }
    };


    const adjustGroupRatiosFromSlider = (changedSlider) => {
        if (isAdjusting) return;
        isAdjusting = true;

        const groupContainer = changedSlider.closest('.ratio-with-slider, .city-ratio-list');
        if (!groupContainer) { isAdjusting = false; return; }

        const isCityGroup = groupContainer.id === 'cityRatioList';
        const groupSelector = isCityGroup ? '.city-ratio-row' : '.field-slider-group';
        const allGroups = Array.from(groupContainer.querySelectorAll(groupSelector));
        const inputs = allGroups.map(g => g.querySelector('.ratio-input')).filter(Boolean);
        const sliders = allGroups.map(g => g.querySelector('.ratio-slider')).filter(Boolean);

        const currentIndex = sliders.indexOf(changedSlider);
        if (currentIndex === -1) { isAdjusting = false; return; }

        // Get value directly from the slider that triggered the event
        const currentValue = clampPercent(parseFloat(changedSlider.value || 0));
        
        // Sync the corresponding number input
        if (inputs[currentIndex]) {
            inputs[currentIndex].value = currentValue;
        }

        const otherInputs = inputs.filter((_, idx) => idx !== currentIndex);
        const otherSliders = sliders.filter((_, idx) => idx !== currentIndex);

        let currentTotal = inputs.reduce((sum, inp) => sum + parseFloat(inp.value || 0), 0);
        let delta = 100 - currentTotal; // Amount needed to balance

        if (Math.abs(delta) < 0.01) { // Already balanced
            updateTotalDisplay(groupContainer); // Still update total display
            isAdjusting = false;
            return;
        }

        // --- Allocation Logic (Only for slider changes) ---
        if (otherInputs.length === 1) { // Two-item group (Gender, MBTI pairs)
            const otherValue = clampPercent(parseFloat(otherInputs[0].value || 0) + delta);
            otherInputs[0].value = otherValue;
            if (otherSliders[0]) otherSliders[0].value = otherValue;
        } else if (otherInputs.length > 1) { // Multi-item group (Frequency, Flavor, Cities)
            let otherTotal = otherInputs.reduce((sum, inp) => sum + parseFloat(inp.value || 0), 0);
            let totalToDistribute = delta;
            let distributedAmount = 0;

            // Filter out items that cannot accept the change (e.g., trying to decrease 0, or increase 100)
             let adjustableInputs = [];
             let adjustableSliders = [];
             let adjustableTotal = 0;

             for(let i=0; i< otherInputs.length; i++){
                 const val = parseFloat(otherInputs[i].value || 0);
                 if ((delta > 0 && val < 100) || (delta < 0 && val > 0)) {
                     adjustableInputs.push(otherInputs[i]);
                     adjustableSliders.push(otherSliders[i]);
                     adjustableTotal += val;
                 }
             }

            if (adjustableInputs.length > 0) {
                // Distribute delta proportionally among adjustable items
                for (let i = 0; i < adjustableInputs.length; i++) {
                    const input = adjustableInputs[i];
                    const slider = adjustableSliders[i];
                    const oldValue = parseFloat(input.value || 0);
                    let share = 0;

                    if (i === adjustableInputs.length - 1) {
                         // Give the last adjustable item the remaining delta to avoid precision errors
                         share = totalToDistribute - distributedAmount;
                    } else {
                        if (delta > 0) { // Increasing others
                             // Distribute increase somewhat evenly or based on remaining capacity (100 - val)?
                             // Simplest: Distribute based on current proportion if otherTotal > 0, else evenly
                              share = adjustableTotal > 0 ? (oldValue / adjustableTotal) * totalToDistribute : totalToDistribute / adjustableInputs.length;
                              // More complex but maybe better: Distribute based on remaining capacity (100 - oldValue)?
                        } else { // Decreasing others (delta is negative)
                            // Distribute decrease proportionally to current value (if possible)
                            share = adjustableTotal > 0 ? (oldValue / adjustableTotal) * totalToDistribute : totalToDistribute / adjustableInputs.length; // Fallback to even distribution
                        }
                    }

                    let newValue = clampPercent(oldValue + share);
                    const actualChange = newValue - oldValue; // How much actually changed after clamping
                    distributedAmount += actualChange;

                    input.value = newValue;
                    if (slider) slider.value = newValue;
                }
                
                // Final precision correction if needed (usually handled by last item getting remainder)
                 let finalSumCheck = inputs.reduce((sum, inp) => sum + parseFloat(inp.value || 0), 0);
                 let finalCorrection = 100 - finalSumCheck;
                  if (Math.abs(finalCorrection) > 0.01 && adjustableInputs.length > 0) {
                      let lastAdjInput = adjustableInputs[adjustableInputs.length - 1];
                      let lastAdjSlider = adjustableSliders[adjustableSliders.length - 1];
                      let correctedLastVal = clampPercent(parseFloat(lastAdjInput.value || 0) + finalCorrection);
                      lastAdjInput.value = correctedLastVal;
                      if(lastAdjSlider) lastAdjSlider.value = correctedLastVal;
                  }

            } else {
                // No other inputs can be adjusted (e.g., all others are 0 and delta is negative)
                 console.warn("Could not fully adjust group total to 100.");
            }
        }
        
        // Ensure all number inputs are formatted nicely after adjustment
         inputs.forEach(inp => inp.value = formatNumber(parseFloat(inp.value || 0), 1));

        updateTotalDisplay(groupContainer);
        isAdjusting = false;
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

const initializeSliders = () => {
        document.querySelectorAll('.ratio-input, .ratio-slider').forEach(input => {
            // 移除旧的 blur 监听器 (如果之前有)
            input.removeEventListener('blur', (event) => {
                 event.target.value = clampPercent(event.target.value || 0);
            });
            // 添加新的联动监听器
            input.addEventListener('input', handleRatioInputChange);
            input.addEventListener('change', handleRatioInputChange); // 确保失焦或回车时也触发
        });
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
        availableCities.forEach(option => { 
            const opt = document.createElement('option');
            opt.value = option;
            opt.textContent = option;
            select.appendChild(opt);
        });
        if (city && availableCities.includes(city)) select.value = city;

        const fieldSliderGroup = document.createElement('div');
        fieldSliderGroup.className = 'field-slider-group';

        const ratioLabel = document.createElement('label');
        ratioLabel.className = 'field';
        const ratioInput = document.createElement('input');
        ratioInput.type = 'number';
        ratioInput.min = '0'; ratioInput.max = '100';
        ratioInput.step = '0.1'; // Allow decimals
        ratioInput.value = clampPercent(ratio ?? 0);
        ratioInput.className = 'ratio-input';
        ratioLabel.appendChild(ratioInput);

        const ratioSlider = document.createElement('input');
        ratioSlider.type = 'range';
        ratioSlider.min = '0'; ratioSlider.max = '100';
        ratioSlider.step = '0.1'; // Match step
        ratioSlider.value = ratioInput.value;
        ratioSlider.className = 'ratio-slider';
        ratioSlider.setAttribute('aria-label', `${city || '城市'}比例滑块`);

        fieldSliderGroup.appendChild(ratioLabel);
        fieldSliderGroup.appendChild(ratioSlider);

        const removeBtn = document.createElement('button');
        removeBtn.type = 'button';
        removeBtn.className = 'remove-city';
        removeBtn.textContent = '移除';
        removeBtn.addEventListener('click', () => {
             if (cityRatioList.querySelectorAll('.city-ratio-row').length <= 1) return;
             row.remove();
             refreshCityRemoveButtons();
             updateAddCityButtonState();
             // 💡 Update total after removing
             updateTotalDisplay(cityRatioList);
         });

        select.addEventListener('change', () => { /* ... update slider label ... */ });

        // Add Listeners
        ratioInput.addEventListener('input', handleRatioInputChange);
        ratioSlider.addEventListener('input', handleRatioInputChange);
        ratioInput.addEventListener('blur', handleNumberInputBlur); // Use blur for final format/clamp

        row.appendChild(select);
        row.appendChild(fieldSliderGroup);
        row.appendChild(removeBtn);
        cityRatioList.appendChild(row);
        refreshCityRemoveButtons();
        updateAddCityButtonState();
    };
    

    const initializeSlidersAndTotals = () => {
        document.querySelectorAll('.ratio-input').forEach(input => {
             input.step = '0.1'; // Allow decimals
             input.addEventListener('input', handleRatioInputChange);
             input.addEventListener('blur', handleNumberInputBlur);
         });
         document.querySelectorAll('.ratio-slider').forEach(slider => {
             slider.step = '0.1'; // Match step
             slider.addEventListener('input', handleRatioInputChange);
         });
         
         // Add total display elements dynamically if they don't exist
         document.querySelectorAll('.ratio-head .hint, .group-head .hint').forEach(hintEl => {
             if (!hintEl.querySelector('.ratio-total-display')) {
                 const totalSpan = document.createElement('span');
                 totalSpan.className = 'ratio-total-display';
                 totalSpan.style.marginLeft = '10px'; // Add some spacing
                 totalSpan.style.fontWeight = 'bold';
                 // Add space before appending if hint has text
                 if(hintEl.textContent.trim().length > 0) hintEl.appendChild(document.createTextNode(' ')); 
                 hintEl.appendChild(totalSpan);
             }
         });

         // Initial total calculation for all groups
         document.querySelectorAll('.ratio-with-slider, .city-ratio-list').forEach(updateTotalDisplay);
    };

const initializeCitySelector = async () => {
        let citiesToUse = ['北京', '上海', '广州', '深圳']; // 默认后备列表
        try {
            const response = await fetch('/get_city_options');
            if (!response.ok) {
                console.error('Failed to fetch cities, status:', response.status);
                throw new Error('Failed to fetch cities');
            }
            const fetchedCities = await response.json();
            if (Array.isArray(fetchedCities) && fetchedCities.length > 0) {
                 availableCities = fetchedCities; // 💡 只有 fetch 成功才更新全局变量
                 citiesToUse = availableCities; // 使用获取到的列表
                 console.log("Successfully fetched cities:", availableCities.length);
            } else {
                 console.warn("Fetched cities list is empty or invalid. Using fallback.");
                 availableCities = ['北京', '上海', '广州', '深圳', '杭州', '成都', '武汉', '南京', '苏州']; // 使用稍长一点的后备
                 citiesToUse = availableCities;
            }

        } catch (error) {
            console.error("Could not initialize city selector via API:", error);
            // 💡 API 请求失败时，使用稍长一点的后备列表填充 availableCities
            availableCities = ['北京', '上海', '广州', '深圳', '杭州', '成都', '武汉', '南京', '苏州']; 
            citiesToUse = availableCities; // 确保 citiesToUse 有值
        } finally {
             if (!cityRatioList) return;
             cityRatioList.innerHTML = ''; // 清空列表

             // 💡 使用 citiesToUse 来决定默认显示的城市
             const defaultCities = ['北京', '上海', '广州', '深圳'];
             const validDefaults = defaultCities.filter(city => citiesToUse.includes(city));

             if (validDefaults.length > 0) {
                  const ratio = 100 / validDefaults.length;
                  validDefaults.forEach(city => createCityRow(city, ratio));
             } else if (citiesToUse.length > 0) {
                 // 如果默认城市不在获取的列表里，用获取列表的前4个
                 const initialCities = citiesToUse.slice(0, 4);
                 const ratio = 100 / initialCities.length;
                 initialCities.forEach(city => createCityRow(city, ratio));
             } else {
                  // 极端情况：连后备列表都空了？
                  console.error("No cities available to display.");
             }
             
             // 确保在所有路径后都更新总和显示
             updateTotalDisplay(cityRatioList);
             updateAddCityButtonState(); // 确保添加按钮状态正确
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
        
        // 1. 界面状态重置
        individualReportsRendered = 0;
        individualContainer.innerHTML = ''; // 清空旧报告
        summaryContainer.innerHTML = '';
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
            // 2. 发送 POST 请求启动任务
            const startResponse = await fetch('/generate_report', { 
                method: 'POST', 
                headers: { 'Content-Type': 'application/json' }, 
                body: JSON.stringify(payload) 
            });
            
            if (!startResponse.ok) throw new Error(`启动分析任务失败: ${startResponse.statusText}`);
            
            const { job_id } = await startResponse.json();
            
            // 3. 💡 关键修改：立即跳转到 Step 3 并启动 CoT 浮窗
            goTo('3');
            startCotStream(job_id); // <--- 启动虚假但真实的思考流

            // 4. 启动真实的报告监听
            const eventSource = new EventSource(`/stream/${job_id}`);
            let realReportStarted = false; // 标记是否已收到真实内容

            eventSource.onmessage = (event) => {
                const message = JSON.parse(event.data);
                
                // --- 💡 修改点 1: 移除之前那个复杂的 if (!realReportStarted...) 判断
                // 我们直接在具体的 case 里控制关闭，这样更精准
                
                switch (message.type) {
                    // --- 🟢 收到整体报告开始信号：立即关闭浮窗 ---
                    case 'summary_report_start':
                        stopCotStream();
                        realReportStarted = true;
                        
                        goTo('4');
                        const container = document.getElementById('summary-report-container');
                        if (container) {
                            container.classList.remove('placeholder-content');
                            // 💡 修改这里：把后面这个 <span> 加回来
                            container.innerHTML = `<div class="prose" id="streaming-summary-text" style="white-space: pre-wrap;"></div><span id="typing-cursor" class="typing-cursor">█</span>`;
                        }
                        break;

                    // --- 🟢 收到文字片段：双重保险 ---
                    case 'summary_report_chunk':
                        if (!realReportStarted) { 
                            stopCotStream(); // <--- 💡 再次保险：万一错过了start信号，收到文字也得关
                            realReportStarted = true; 
                        }
                        
                        const textEl = document.getElementById('streaming-summary-text');
                        if (textEl) {
                            textEl.textContent += message.data;
                            // 可选：自动滚动
                            // textEl.scrollIntoView({ behavior: "smooth", block: "end" });
                        }
                        break;

                    case 'individual_report':
                        // 如果之前跳过了整体分析直接发个人报告（虽然现在逻辑不会），也得关
                        if (!realReportStarted) { stopCotStream(); realReportStarted = true; }

                        if (individualReportsRendered === 0) {
                            individualContainer.innerHTML = ''; 
                        }
                        displayIndividualReport(message.data);
                        individualReportsRendered++;
                        
                        const existingLoader = document.getElementById('next-report-loader');
                        if (existingLoader) existingLoader.remove();
                        
                        if (individualReportsRendered < personas.length) {
                            individualContainer.insertAdjacentHTML('beforeend', `<div id="next-report-loader" class="persona-report-loading"><div class="spinner-small"></div><span>正在生成下一个画像分析报告...</span></div>`);
                        } else {
                            nextToSummaryContainer.style.display = 'flex';
                        }
                        break;
                    
                    case 'chart_and_table': 
                        // 图表出来时，移除打字光标
                        document.getElementById('typing-cursor')?.remove();
                        displayChartAndTable(message.data); 
                        break;
                    
                    case 'table_analysis': 
                        displayTableAnalysis(message.data); 
                        break;
                    
                    case 'done': 
                        eventSource.close();
                        document.getElementById('next-report-loader')?.remove();
                        document.getElementById('typing-cursor')?.remove();
                        stopCotStream(); // 最终保险
                        break;
                    
                    case 'error':
                        stopCotStream(); // 出错立即关
                        individualContainer.innerHTML = `<div class="error-message">分析出错: ${message.data}</div>`;
                        eventSource.close();
                        break;
                }
            };
            
            eventSource.onerror = (err) => {
                console.error("数据流连接错误:", err);
                stopCotStream();
                individualContainer.innerHTML = `<div class="error-message">与服务器的连接丢失。</div>`;
                eventSource.close();
            };

        } catch (error) {
            console.error("无法启动分析任务:", error);
            stopCotStream();
            alert(`无法启动分析任务: ${error.message}`);
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

brandSelect?.addEventListener('change', (e) => {
      const selectedBrand = e.target.value;
      
      if (selectedBrand === 'polyu') {
        brandPolyu?.classList.remove('hidden');
        brandMingyin?.classList.add('hidden');
      } else {
        brandPolyu?.classList.add('hidden');
        brandMingyin?.classList.remove('hidden');
      }
    });

addCityRatioBtn?.addEventListener('click', () => {
        const used = new Set(Array.from(cityRatioList.querySelectorAll('select')).map(sel => sel.value));
        const nextCity = availableCities.find(city => !used.has(city));
        if (nextCity) {
            createCityRow(nextCity, 0); // Add new row with 0
            // 💡 Update total display after adding
             updateTotalDisplay(cityRatioList);
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
    initializeSlidersAndTotals();
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