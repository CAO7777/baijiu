// static/script.js (v4.6 - Final, Unabridged Version)

document.addEventListener('DOMContentLoaded', () => {
    // --- 1. STATE MANAGEMENT ---
    let personas = [];
    let currentPersona = null;
    let individualReportsRendered = 0;
    let productImageDataURL = null;

    // --- 2. DOM ELEMENT SELECTORS ---
    const steps = document.querySelectorAll('.panel.step');
    const pills = document.querySelectorAll('.step-pill');
    const personaForm = document.getElementById('persona-form');
    const genderEl = document.getElementById('gender');
    const ageEl = document.getElementById('age');
    const cityEl = document.getElementById('city');
    const professionEl = document.getElementById('profession');
    const eduEl = document.getElementById('edu');
    const incomeEl = document.getElementById('income');
    const priceEl = document.getElementById('price');
    const drinkFreqEl = document.getElementById('drinkFreq');
    const drinkYearsEl = document.getElementById('drinkYears');
    const flavorEl = document.getElementById('flavor');
    const mbtiRadios = {
        energy: document.querySelectorAll('input[name="energy"]'),
        info: document.querySelectorAll('input[name="info"]'),
        decision: document.querySelectorAll('input[name="decision"]'),
        life: document.querySelectorAll('input[name="life"]'),
    };
    const addPersonaBtn = document.getElementById('addPersona');
    const personaTagsContainer = document.getElementById('personaTags');
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
        if (targetStepEl) targetStepEl.classList.remove('hidden');
        pills.forEach(p => p.classList.toggle('active', p.dataset.step == step));
        window.scrollTo({ top: 0, behavior: 'smooth' });
    };

    // --- 4. PERSONA MANAGEMENT LOGIC ---
    const collectPersonaPayload = () => {
        const getRadioValue = (radios) => (Array.from(radios).find(r => r.checked)?.value || '');
        const mbti = `${getRadioValue(mbtiRadios.energy)}${getRadioValue(mbtiRadios.info)}${getRadioValue(mbtiRadios.decision)}${getRadioValue(mbtiRadios.life)}`;
        return {
            gender: genderEl.value, age: ageEl.value, city: cityEl.value.trim(),
            profession: professionEl.value.trim(), mbti,
            education: eduEl.value, income: incomeEl.value,
            expected_price: priceEl.value, drink_frequency: drinkFreqEl.value,
            drinking_history: drinkYearsEl.value.trim(), preferred_aroma: flavorEl.value,
        };
    };

    const renderPersonaTags = () => {
        personaTagsContainer.innerHTML = '';
        personas.forEach((p, index) => {
            const el = document.createElement('span');
            el.className = 'tag';
            el.textContent = `${p.age}岁 ${p.gender} · ${p.city || '未知城市'} · MBTI: ${p.mbti}`;
            el.dataset.index = index;
            el.title = "点击修改此画像";
            el.addEventListener('click', () => loadPersonaForEdit(index));
            personaTagsContainer.appendChild(el);
        });
        if (!currentPersona) {
            personaForm.reset();
            ageEl.value = '30';
            document.querySelector('input[name="energy"][value="I"]').checked = true;
            document.querySelector('input[name="info"][value="S"]').checked = true;
            document.querySelector('input[name="decision"][value="T"]').checked = true;
            document.querySelector('input[name="life"][value="J"]').checked = true;
        }
    };

    const loadPersonaForEdit = (index) => {
        currentPersona = personas[index];
        genderEl.value = currentPersona.gender; ageEl.value = currentPersona.age;
        cityEl.value = currentPersona.city; professionEl.value = currentPersona.profession;
        eduEl.value = currentPersona.education || ''; incomeEl.value = currentPersona.income || '';
        priceEl.value = currentPersona.expected_price || ''; drinkFreqEl.value = currentPersona.drink_frequency || '';
        drinkYearsEl.value = currentPersona.drinking_history || ''; flavorEl.value = currentPersona.preferred_aroma || '';
        
        const mbtiChars = (currentPersona.mbti || "ISTJ").split('');
        if(mbtiChars[0]) document.querySelector(`input[name="energy"][value="${mbtiChars[0]}"]`).checked = true;
        if(mbtiChars[1]) document.querySelector(`input[name="info"][value="${mbtiChars[1]}"]`).checked = true;
        if(mbtiChars[2]) document.querySelector(`input[name="decision"][value="${mbtiChars[2]}"]`).checked = true;
        if(mbtiChars[3]) document.querySelector(`input[name="life"][value="${mbtiChars[3]}"]`).checked = true;
        
        addPersonaBtn.textContent = '保存修改';
        addPersonaBtn.classList.remove('outline'); addPersonaBtn.classList.add('primary');
        let cancelBtn = document.getElementById('cancelEditPersona');
        if (!cancelBtn) {
            cancelBtn = document.createElement('button');
            cancelBtn.id = 'cancelEditPersona'; cancelBtn.type = 'button';
            cancelBtn.className = 'btn outline'; cancelBtn.textContent = '取消修改';
            addPersonaBtn.after(cancelBtn);
            cancelBtn.addEventListener('click', cancelEditPersona);
        }
    };

    const cancelEditPersona = () => {
        currentPersona = null;
        addPersonaBtn.textContent = '添加新画像';
        addPersonaBtn.classList.add('outline'); addPersonaBtn.classList.remove('primary');
        document.getElementById('cancelEditPersona')?.remove();
        renderPersonaTags();
    };

    const addOrUpdatePersona = () => {
        if (currentPersona) {
            const updatedPersona = collectPersonaPayload();
            const indexToUpdate = personas.indexOf(currentPersona);
            if (indexToUpdate !== -1) personas[indexToUpdate] = updatedPersona;
            cancelEditPersona();
        } else {
            personas.push(collectPersonaPayload());
            renderPersonaTags();
        }
    };

    // --- 5. PRODUCT INFO LOGIC ---
    const handleImageUpload = (event) => {
        productPreviewEl.innerHTML = '';
        const file = event.target.files[0];
        if (!file) {
            productPreviewEl.classList.add('tip');
            productPreviewEl.textContent = '建议上传横向高清图，展示瓶身与包装细节。';
            productImageDataURL = null;
            return;
        };
        const reader = new FileReader();
        reader.onload = () => {
            productImageDataURL = reader.result;
            const img = document.createElement('img');
            img.src = productImageDataURL;
            img.alt = '产品图片预览';
            productPreviewEl.classList.remove('tip');
            productPreviewEl.appendChild(img);
        };
        reader.readAsDataURL(file);
    };

    // --- 6. REPORT GENERATION & STREAMING LOGIC ---
    const generateReport = async () => {
        if (personas.length === 0) { alert('请至少添加一个画像。'); return; }
        if (!productDescEl.value.trim() || !productImageDataURL) { alert('请完善产品描述并上传产品图片。'); return; }
        
        goTo('3');
        
        individualReportsRendered = 0;
        summaryContainer.innerHTML = '<div class="placeholder-content"><div class="spinner"></div><p>等待独立报告生成完毕...</p></div>';
        chartsContainer.innerHTML = '';
        nextToSummaryContainer.style.display = 'none';

        const nextReportLoaderHTML = `<div id="next-report-loader" class="persona-report-loading"><div class="spinner-small"></div><span>正在生成下一个分析报告...</span></div>`;
        individualContainer.innerHTML = nextReportLoaderHTML;

        const payload = { personas, productData: { description: productDescEl.value, image: productImageDataURL } };

        try {
            const startResponse = await fetch('/generate_report', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
            if (!startResponse.ok) throw new Error(`启动分析任务失败: ${startResponse.statusText}`);
            const { job_id } = await startResponse.json();
            const eventSource = new EventSource(`/stream/${job_id}`);

            eventSource.onmessage = (event) => {
                const message = JSON.parse(event.data);
                
                switch (message.type) {
                    case 'individual_report':
                        const reportEl = displayIndividualReport(message.data);
                        const loader = document.getElementById('next-report-loader');
                        if (individualReportsRendered === 0 && loader) {
                            // If it's the first report, replace the loader entirely
                            loader.replaceWith(reportEl);
                        } else if (loader) {
                            // For subsequent reports, insert before the loader
                            individualContainer.insertBefore(reportEl, loader);
                        } else {
                            // Fallback if loader is not found
                            individualContainer.appendChild(reportEl);
                        }
                        individualReportsRendered++;
                        if (individualReportsRendered === personas.length) {
                            document.getElementById('next-report-loader')?.remove();
                            nextToSummaryContainer.style.display = 'flex';
                        }
                        break;
                    case 'summary_report': displaySummaryReport(message.data); break;
                    case 'chart_and_table': displayChartAndTable(message.data); break;
                    case 'table_analysis': displayTableAnalysis(message.data); break;
                    case 'done':
                        eventSource.close();
                        document.getElementById('next-report-loader')?.remove();
                        break;
                    case 'error':
                        individualContainer.innerHTML = `<div class="error-message">分析出错: ${message.data}</div>`;
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
const renderRadarChart = (canvasId, scores) => {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    const labels = Object.keys(scores);
    const dataPoints = Object.values(scores);

    new Chart(ctx, {
        type: 'radar',
        data: {
            labels: labels,
            datasets: [{
                label: '匹配度分数',
                data: dataPoints,
                fill: true,
                backgroundColor: 'rgba(110, 15, 26, 0.25)',
                borderColor: 'rgb(110, 15, 26)',
                pointBackgroundColor: 'rgb(110, 15, 26)',
                pointBorderColor: '#fff',
                pointHoverBackgroundColor: '#fff',
                pointHoverBorderColor: 'rgb(110, 15, 26)'
            }]
        },
        options: {
            responsive: false,
            maintainAspectRatio: false,
            scales: {
                r: {
                    
                    startAngle: 0,
                    
                    angleLines: { color: '#E9E3DA' },
                    grid: { circular: true, color: '#E9E3DA' },
                    pointLabels: { font: { size: 14, family: "'Noto Serif SC', serif" }, color: '#2A1B1B' },
                    ticks: {
                        display: true,
                        color: '#6E0F1A',
                        font: { weight: 'bold' },
                        backdropColor: 'rgba(250, 248, 245, 0.75)',
                        backdropPadding: 2,
                        stepSize: 2
                    },
                    min: 0,
                    max: 10
                }
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `${context.label}: ${context.raw} / 10`;
                        }
                    }
                }
            }
        }
    });
};
    
const displayIndividualReport = (data) => {
    const reportEl = document.createElement('div');
    reportEl.className = 'persona-report';
    
    const persona = data.persona_details;
    const cityText = persona.city ? `，来自${persona.city}` : '';
    const decisionClass = data.decision === '购买' ? 'decision-buy' : 'decision-nobuy';
    
    const reportContent = data.report || {};
    const packagingAnalysis = reportContent.packaging_analysis || "AI未提供此项分析。";
    const fitAnalysis = reportContent.fit_analysis || "AI未提供此项分析。";
    const scenarioAnalysis = reportContent.scenario_analysis || "AI未提供此项分析。";
    const finalDecision = data.final_decision || {};
    const radarCanvasId = `radar-chart-${data.persona_id}`;

    reportEl.innerHTML = `
        <h4 class="h4">画像 ${data.persona_id} 独立报告</h4>
        <p class="muted">画像简介：${persona.age}岁 ${persona.gender}${cityText}，MBTI: ${persona.mbti}。</p>
        
        <div class="report-details-grid">
            <div class="report-text-sections">
                <div class="report-section"><h5>包装视觉评估</h5><p>${packagingAnalysis}</p></div>
                <div class="report-section"><h5>产品契合度分析</h5><p>${fitAnalysis}</p></div>
                <div class="report-section"><h5>潜在消费场景</h5><p>${scenarioAnalysis}</p></div>
            </div>
            <div class="radar-chart-container">
                <h5>画像-产品匹配度</h5>
                
                <canvas id="${radarCanvasId}" width="400" height="400"></canvas>

            </div>
        </div>

        <div class="final-decision-section">
            <p><strong class="${decisionClass}">【最终决策】</strong>${finalDecision.decision || '未明确'}</p>
            <p><strong>【决策理由】</strong>${finalDecision.reason || 'AI未提供理由。'}</p>
        </div>
    `;
    
    if (data.radar_scores && Object.keys(data.radar_scores).length > 0) {
        setTimeout(() => {
            renderRadarChart(radarCanvasId, data.radar_scores);
        }, 0);
    }
    return reportEl; 
};
    
    const displaySummaryReport = (data) => {
        summaryContainer.classList.remove('placeholder-content');
        summaryContainer.innerHTML = `<p>${data.replace(/\n\n/g, '</p><p>').replace(/\n/g, '<br>')}</p>`;
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
            </div>
        `;
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
                alert('请至少添加一个画像再进行下一步。');
                return;
            }
            if (e.currentTarget.id === 'genReport') return; 
            goTo(targetStep);
        });
    });

    addPersonaBtn.addEventListener('click', addOrUpdatePersona);
    productImgInput.addEventListener('change', handleImageUpload);
    genReportBtn.addEventListener('click', generateReport);
    exportBtn.addEventListener('click', () => alert('导出功能待实现。'));

    // Initialize
    renderPersonaTags();
});