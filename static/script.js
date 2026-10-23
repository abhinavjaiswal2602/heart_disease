// Preset configurations
const PRESETS = {
    low: {
        age: 32, sex: "0", cp: "3", exang: "0",
        trestbps: 110, chol: 175, thalach: 172, fbs: "0",
        restecg: "0", oldpeak: 0.0, slope: "1", ca: 0, thal: "3"
    },
    high: {
        age: 67, sex: "1", cp: "4", exang: "1",
        trestbps: 120, chol: 229, thalach: 129, fbs: "0",
        restecg: "2", oldpeak: 2.6, slope: "2", ca: 2, thal: "7"
    },
    borderline: {
        age: 55, sex: "1", cp: "2", exang: "0",
        trestbps: 135, chol: 240, thalach: 140, fbs: "1",
        restecg: "0", oldpeak: 1.2, slope: "2", ca: 1, thal: "3"
    }
};

// Model benchmark metadata from project validation report
const MODEL_BENCHMARKS = {
    'K-Nearest Neighbors': {
        recommended: true,
        badgeText: 'Top Recall (Primary)',
        meta: 'Recall: 82.1% • Accuracy: 85.0% • ROC-AUC: 0.915'
    },
    'Random Forest': {
        recommended: false,
        badgeText: 'Ensemble Tree',
        meta: 'Recall: 78.6% • Accuracy: 83.3% • ROC-AUC: 0.939'
    },
    'Logistic Regression': {
        recommended: false,
        badgeText: 'Linear Baseline',
        meta: 'Recall: 75.0% • Accuracy: 83.3% • ROC-AUC: 0.944'
    },
    'Decision Tree': {
        recommended: false,
        badgeText: 'Decision Tree',
        meta: 'Recall: 50.0% • Accuracy: 71.7% • ROC-AUC: 0.796'
    }
};

// Preset button handlers
document.querySelectorAll('.preset-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        const type = btn.getAttribute('data-preset');
        const data = PRESETS[type];
        if (!data) return;

        // Fill form fields
        for (const [key, value] of Object.entries(data)) {
            const input = document.getElementById(`input-${key}`);
            if (input) {
                input.value = value;
            }
        }

        document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');

        // Auto-submit to update diagnostics
        document.getElementById('prediction-form').requestSubmit();
    });
});

// Form submission handler
document.getElementById('prediction-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const form = e.target;
    const submitBtn = document.getElementById('submit-btn');
    const btnText = submitBtn.querySelector('.btn-text');
    const spinner = submitBtn.querySelector('.spinner');
    
    const formData = new FormData(form);
    const data = {};
    for (let [key, value] of formData.entries()) {
        data[key] = parseFloat(value);
    }
    
    // UI state: loading
    btnText.classList.add('hidden');
    spinner.classList.remove('hidden');
    submitBtn.disabled = true;
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data),
        });
        
        const result = await response.json();
        
        if (response.ok) {
            renderResults(result, data);
        } else {
            alert('Error: ' + (result.error || 'Evaluation failed.'));
        }
    } catch (error) {
        alert('Unable to reach the diagnostic server.');
    } finally {
        btnText.classList.remove('hidden');
        spinner.classList.add('hidden');
        submitBtn.disabled = false;
    }
});

function renderResults(results, patientInput) {
    const welcomeEl = document.getElementById('welcome-state');
    const resultsEl = document.getElementById('results-state');
    if (welcomeEl) welcomeEl.classList.add('hidden');
    if (resultsEl) resultsEl.classList.remove('hidden');
    
    const modelNames = Object.keys(results);
    let totalRiskCount = 0;
    let sumProbability = 0;

    // 1. Calculate ensemble metrics
    modelNames.forEach(name => {
        if (results[name].prediction === 1) totalRiskCount++;
        sumProbability += results[name].probability;
    });

    const avgProbability = (sumProbability / modelNames.length) * 100;
    
    // 2. Update Gauge Visual
    updateRiskGauge(avgProbability, totalRiskCount, modelNames.length);

    // 3. Update Biomarker Assessment Pills
    renderBiomarkers(patientInput);

    // 4. Render individual model cards with 50% threshold guideline
    const container = document.getElementById('models-container');
    container.innerHTML = '';
    
    modelNames.forEach(name => {
        const data = results[name];
        const isHigh = data.prediction === 1;
        const prob = (data.probability * 100).toFixed(1);
        const metaInfo = MODEL_BENCHMARKS[name] || { recommended: false, badgeText: '', meta: '' };

        const card = document.createElement('div');
        card.className = `model-card ${isHigh ? 'risk-high' : 'risk-low'}`;
        card.innerHTML = `
            <div class="model-header">
                <div class="model-title-wrap">
                    <span class="model-name">${name}</span>
                    ${metaInfo.recommended ? `<span class="model-rec-chip">${metaInfo.badgeText}</span>` : ''}
                </div>
                <div class="model-pred ${isHigh ? 'pred-high' : 'pred-low'}">
                    ${isHigh ? 'Positive (High Risk)' : 'Negative (Normal)'}
                </div>
            </div>
            <div class="model-meta">${metaInfo.meta}</div>
            
            <div class="prob-meter-wrap">
                <div class="prob-meter">
                    <div class="prob-threshold-mark" title="50% Clinical Decision Boundary"></div>
                    <div class="prob-fill ${isHigh ? 'fill-high' : 'fill-low'}" style="width: ${prob}%;"></div>
                </div>
            </div>
            
            <div class="prob-text">
                <span>Calculated Probability: <strong>${prob}%</strong></span>
                <span class="cutoff-tag">50.0% Decision Cutoff</span>
            </div>
        `;
        container.appendChild(card);
    });
    
    // 5. Handle Consensus Badge, Disagreement Warning, and Clinical Verdict
    const badge = document.getElementById('consensus-badge');
    const alertBox = document.getElementById('disagreement-alert');
    const verdictPanel = document.getElementById('verdict-panel');
    
    if (totalRiskCount > 0 && totalRiskCount < modelNames.length) {
        // Disagreement exists
        if (alertBox) alertBox.classList.remove('hidden');
        if (badge) {
            badge.className = 'badge warning';
            badge.textContent = `Mixed Consensus (${totalRiskCount}/${modelNames.length} Positive)`;
        }
        
        if (verdictPanel) {
            verdictPanel.innerHTML = `
                <div class="verdict-banner banner-warning">
                    <div class="verdict-title-row">
                        <div>
                            <h3 class="verdict-title text-warning">Clinical Review Advised — Discrepant Model Classifications</h3>
                            <span class="verdict-meta">Ensemble Probability: ${avgProbability.toFixed(1)}% • ${totalRiskCount} of ${modelNames.length} models flag cardiac disease markers</span>
                        </div>
                    </div>
                    
                    <p class="verdict-desc">
                        The models produced differing predictions for this patient. In screening environments, false negatives represent the highest clinical risk. Because <strong>K-Nearest Neighbors</strong> achieved the highest test recall (82.1%), triage should lean conservative with secondary diagnostic testing.
                    </p>

                    <div class="clinical-action-box">
                        <div class="action-box-title">Recommended Clinical Steps:</div>
                        <ul class="action-list">
                            <li><strong>Secondary Diagnostic:</strong> Order stress echocardiography or coronary CTA.</li>
                            <li><strong>Lab Follow-up:</strong> Repeat fasting lipid profile and consider hs-CRP assessment.</li>
                            <li><strong>Monitoring:</strong> Consider 24-hour ambulatory Holter ECG monitoring.</li>
                        </ul>
                    </div>
                </div>
            `;
        }
    } else {
        if (alertBox) alertBox.classList.add('hidden');
        
        if (totalRiskCount === 0) {
            if (badge) {
                badge.className = 'badge safe';
                badge.textContent = 'Unanimous Agreement (Low Risk)';
            }
            
            if (verdictPanel) {
                verdictPanel.innerHTML = `
                    <div class="verdict-banner banner-safe">
                        <div class="verdict-title-row">
                            <div>
                                <h3 class="verdict-title text-safe">Negative Screening — Low Cardiovascular Risk</h3>
                                <span class="verdict-meta">Ensemble Probability: ${avgProbability.toFixed(1)}% • All 4 Classifiers Agree</span>
                            </div>
                        </div>

                        <p class="verdict-desc">
                            All four classifiers independently evaluated this patient profile with no significant indicators of coronary heart disease. Vitals, perfusion markers, and ST-segment measurements fall within expected non-ischemic ranges.
                        </p>

                        <div class="clinical-action-box">
                            <div class="action-box-title">Preventative Care Summary:</div>
                            <ul class="action-list">
                                <li><strong>Routine Follow-up:</strong> Standard periodic health maintenance.</li>
                                <li><strong>Lifestyle:</strong> Maintain regular physical activity and balanced cardiovascular nutrition.</li>
                                <li><strong>Vascular Health:</strong> Continue routine blood pressure and lipid monitoring.</li>
                            </ul>
                        </div>
                    </div>
                `;
            }
        } else {
            if (badge) {
                badge.className = 'badge danger';
                badge.textContent = 'Unanimous Critical Alert (High Risk)';
            }
            
            if (verdictPanel) {
                verdictPanel.innerHTML = `
                    <div class="verdict-banner banner-danger">
                        <div class="verdict-title-row">
                            <div>
                                <h3 class="verdict-title text-danger">Critical Alert — High Probability of Ischemic Heart Disease</h3>
                                <span class="verdict-meta">Ensemble Probability: ${avgProbability.toFixed(1)}% • All 4 Classifiers Agree</span>
                            </div>
                        </div>

                        <p class="verdict-desc">
                            All four classifiers unanimously identified markers indicative of coronary heart disease. The presence of abnormal ST depression, exercise-induced symptoms, or fluoroscopy vessel findings suggests significant likelihood of myocardial ischemia.
                        </p>

                        <div class="clinical-action-box">
                            <div class="action-box-title">Recommended Immediate Actions:</div>
                            <ul class="action-list">
                                <li><strong>Specialist Referral:</strong> Urgent cardiology consultation for catheterization/angiography review.</li>
                                <li><strong>Therapy Review:</strong> Evaluate guideline-directed medical therapy.</li>
                                <li><strong>Activity Guidance:</strong> Restrict strenuous exertion pending definitive clinical evaluation.</li>
                            </ul>
                        </div>
                    </div>
                `;
            }
        }
    }
}

// Function to update Semi-Circle Gauge Visual
function updateRiskGauge(avgProb, riskCount, totalModels) {
    const gaugeFill = document.getElementById('gauge-fill');
    const gaugeVal = document.getElementById('gauge-val');
    const gaugeBadge = document.getElementById('gauge-badge');
    const gaugeCaption = document.getElementById('gauge-caption');

    const circumference = 188.5; // pi * 60
    const clampedProb = Math.min(Math.max(avgProb, 0), 100);
    const strokeOffset = circumference * (1 - clampedProb / 100);

    if (gaugeFill) {
        gaugeFill.style.strokeDashoffset = strokeOffset.toFixed(1);
    }
    if (gaugeVal) {
        gaugeVal.textContent = `${clampedProb.toFixed(0)}%`;
    }

    if (clampedProb < 35) {
        if (gaugeFill) gaugeFill.style.stroke = '#16a34a';
        if (gaugeBadge) {
            gaugeBadge.className = 'gauge-badge safe';
            gaugeBadge.textContent = 'Low Risk';
        }
    } else if (clampedProb <= 65) {
        if (gaugeFill) gaugeFill.style.stroke = '#d97706';
        if (gaugeBadge) {
            gaugeBadge.className = 'gauge-badge warning';
            gaugeBadge.textContent = 'Borderline';
        }
    } else {
        if (gaugeFill) gaugeFill.style.stroke = '#dc2626';
        if (gaugeBadge) {
            gaugeBadge.className = 'gauge-badge danger';
            gaugeBadge.textContent = 'High Risk';
        }
    }

    if (gaugeCaption) {
        gaugeCaption.textContent = `${riskCount} of ${totalModels} classifiers flagged risk`;
    }
}

// Function to render patient biomarker flags with clean medical styling
function renderBiomarkers(input) {
    const grid = document.getElementById('biomarkers-grid');
    if (!grid) return;
    grid.innerHTML = '';

    const biomarkers = [
        {
            label: 'Blood Pressure',
            val: `${input.trestbps} mmHg`,
            status: input.trestbps < 120 ? 'Normal (<120)' : input.trestbps < 140 ? 'Elevated (120-139)' : 'Stage 2 (≥140)',
            class: input.trestbps < 120 ? 'status-safe' : input.trestbps < 140 ? 'status-warning' : 'status-danger'
        },
        {
            label: 'Serum Cholesterol',
            val: `${input.chol} mg/dL`,
            status: input.chol < 200 ? 'Desirable (<200)' : input.chol < 240 ? 'Borderline (200-239)' : 'High (≥240)',
            class: input.chol < 200 ? 'status-safe' : input.chol < 240 ? 'status-warning' : 'status-danger'
        },
        {
            label: 'Max Heart Rate',
            val: `${input.thalach} bpm`,
            status: input.thalach >= 140 ? 'Normal (≥140)' : 'Depressed (<140)',
            class: input.thalach >= 140 ? 'status-safe' : 'status-danger'
        },
        {
            label: 'ST Depression',
            val: `${input.oldpeak} mm`,
            status: input.oldpeak <= 0.5 ? 'Normal (≤0.5)' : input.oldpeak <= 1.5 ? 'Mild (0.6-1.5)' : 'Severe (>1.5)',
            class: input.oldpeak <= 0.5 ? 'status-safe' : input.oldpeak <= 1.5 ? 'status-warning' : 'status-danger'
        },
        {
            label: 'Major Vessels (ca)',
            val: `${input.ca} colored`,
            status: input.ca === 0 ? 'Clear (0)' : `${input.ca} Narrowed`,
            class: input.ca === 0 ? 'status-safe' : 'status-danger'
        },
        {
            label: 'Exercise Angina',
            val: input.exang === 1 ? 'Positive' : 'Negative',
            status: input.exang === 1 ? 'Pain on Exertion' : 'No Pain',
            class: input.exang === 1 ? 'status-danger' : 'status-safe'
        }
    ];

    biomarkers.forEach(b => {
        const item = document.createElement('div');
        item.className = 'biomarker-pill';
        item.innerHTML = `
            <span class="biomarker-label">${b.label}</span>
            <div class="biomarker-row">
                <span class="biomarker-val">${b.val}</span>
                <span class="biomarker-status ${b.class}">${b.status}</span>
            </div>
        `;
        grid.appendChild(item);
    });
}

// Research Plots Modal logic
const plotsModal = document.getElementById('plots-modal');
const openPlotsBtn = document.getElementById('open-plots-btn');
const closePlotsBtn = document.getElementById('close-plots-btn');
const plotImg = document.getElementById('plot-image');
const plotCaption = document.getElementById('plot-caption');
const tabButtons = document.querySelectorAll('.modal-tabs .tab-btn');

if (openPlotsBtn && plotsModal) {
    openPlotsBtn.addEventListener('click', () => {
        plotsModal.classList.remove('hidden');
    });

    closePlotsBtn.addEventListener('click', () => {
        plotsModal.classList.add('hidden');
    });

    plotsModal.querySelector('.modal-backdrop').addEventListener('click', () => {
        plotsModal.classList.add('hidden');
    });

    window.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && !plotsModal.classList.contains('hidden')) {
            plotsModal.classList.add('hidden');
        }
    });

    tabButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            tabButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            const imgSrc = btn.getAttribute('data-img');
            const desc = btn.getAttribute('data-desc');

            if (plotImg) plotImg.src = imgSrc;
            if (plotCaption) plotCaption.textContent = desc;
        });
    });
}
