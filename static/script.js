/* 
dg  -> data generation
dgs -> data generation summary
dgd -> data generation dropdown
mh  -> model hyperparameters
mhs -> model hyperparameters summary
mhd -> model hyperparameters dropdown
lr  -> logistic regression
dt  -> decision tree

// const MH_COLOR = "rgba(255,100,130, {{alpha}})";

We gotta add `new Date().getTime()` so that the image is actually updated
*/



// Get the value of the variables in :root in style.css
const BORDER_RADIUS  = getComputedStyle(document.documentElement).getPropertyValue('--border-radius');
const DG_COLOR       = getComputedStyle(document.documentElement).getPropertyValue('--dg-color');
const MH_COLOR       = getComputedStyle(document.documentElement).getPropertyValue('--mh-color');
const ACTIVE_ALPHA   = getComputedStyle(document.documentElement).getPropertyValue('--active-alpha');
const INACTIVE_ALPHA = getComputedStyle(document.documentElement).getPropertyValue('--inactive-alpha');

// Variables
let is_opened_dgd = false;
let is_opened_mhd = false;

// Elements
const dgs                                   = document.getElementById('dgs');
const mhs                                   = document.getElementById('mhs');
const dgd                                   = document.getElementById('dgd');
const mhd                                   = document.getElementById('mhd');
const dg_template                           = document.getElementById('dgd-template');
const dg_noise                              = document.getElementById('dgd-noise');
const dg_factor                             = document.getElementById('dgd-factor');
const dg_n_classes                          = document.getElementById('dgd-n-classes');
const dg_class_sep                          = document.getElementById('dgd-class-sep');
const mh_svm_degree                         = document.getElementById('mhd-svm-degree');
const mh_svm_gamma                          = document.getElementById('mhd-svm-gamma');
const mh_lr_l1_ratio                        = document.getElementById('mhd-lr-l1-ratio');
const mh_dt_max_depth_int                   = document.getElementById('mhd-dt-max-depth-int');
const mh_dt_max_features_int                = document.getElementById('mhd-dt-max-features-int');
const mh_dt_max_features_float              = document.getElementById('mhd-dt-max-features-float');
const cmap_input                            = document.getElementById('dgd-cmap-input');
const template_input                        = document.getElementById('dgd-template-input');
const n_samples_input                       = document.getElementById('dgd-n-samples-input');
const noise_input                           = document.getElementById('dgd-noise-input');
const factor_input                          = document.getElementById('dgd-factor-input');
const n_classes_input                       = document.getElementById('dgd-n-classes-input');
const class_sep_input                       = document.getElementById('dgd-class-sep-input');
const mhd_knn_k_input                       = document.getElementById('mhd-knn-k-input');
const mhd_knn_distance_metric_input         = document.getElementById('mhd-knn-distance-metric-input');
const mhd_svm_kernel_input                  = document.getElementById('mhd-svm-kernel-input');
const mhd_svm_c_input                       = document.getElementById('mhd-svm-c-input');
const mhd_svm_gamma_input                   = document.getElementById('mhd-svm-gamma-input');
const mhd_svm_gamma_float_input             = document.getElementById('mhd-svm-gamma-float-input');
const mhd_svm_degree_input                  = document.getElementById('mhd-svm-degree-input');
const mhd_lr_penalty_input                  = document.getElementById('mhd-lr-penalty-input');
const mhd_lr_c_input                        = document.getElementById('mhd-lr-c-input');
const mhd_lr_l1_ratio_input                 = document.getElementById('mhd-lr-l1-ratio-input');
const mhd_lr_l1_ratio_float_input           = document.getElementById('mhd-lr-l1-ratio-float-input');
const mhd_dt_criterion_input                = document.getElementById('mhd-dt-criterion-input');
const mhd_dt_splitter_input                 = document.getElementById('mhd-dt-splitter-input');
const mhd_dt_max_depth_input                = document.getElementById('mhd-dt-max-depth-input');
const mhd_dt_max_depth_int_input            = document.getElementById('mhd-dt-max-depth-int-input');
const mhd_dt_max_features_input             = document.getElementById('mhd-dt-max-features-input');
const mhd_dt_max_features_int_input         = document.getElementById('mhd-dt-max-features-int-input');
const mhd_dt_max_features_float_input       = document.getElementById('mhd-dt-max-features-float-input');
const mhd_ensemble_n_estimators_input       = document.getElementById('mhd-ensemble-n-estimators-input');
const mhd_ensemble_criterion_input          = document.getElementById('mhd-ensemble-criterion-input');
const mhd_ensemble_max_depth_input          = document.getElementById('mhd-ensemble-max-depth-input');
const mhd_ensemble_max_depth_int_input      = document.getElementById('mhd-ensemble-max-depth-int-input');
const mhd_ensemble_max_features_input       = document.getElementById('mhd-ensemble-max-features-input');
const mhd_ensemble_max_features_int_input   = document.getElementById('mhd-ensemble-max-features-int-input');
const mhd_ensemble_max_features_float_input = document.getElementById('mhd-ensemble-max-features-float-input');
const plot_unclassified                     = document.getElementById('plot-unclassified');
const plot_knn                              = document.getElementById('plot-knn');
const plot_svm                              = document.getElementById('plot-svm');
const plot_lr                               = document.getElementById('plot-lr');
const plot_dt                               = document.getElementById('plot-dt');
const plot_rf                               = document.getElementById('plot-rf');
const plot_et                               = document.getElementById('plot-et');



// Change bg colors
dgs.style.backgroundColor = DG_COLOR.replace("{{alpha}}", INACTIVE_ALPHA);
dgd.style.backgroundColor = DG_COLOR.replace("{{alpha}}", INACTIVE_ALPHA);
mhs.style.backgroundColor = MH_COLOR.replace("{{alpha}}", INACTIVE_ALPHA);
mhd.style.backgroundColor = MH_COLOR.replace("{{alpha}}", INACTIVE_ALPHA);
// Change border radius-es
dgs.style.borderRadius            = BORDER_RADIUS;
mhs.style.borderRadius            = BORDER_RADIUS;
dgd.style.borderBottomLeftRadius  = BORDER_RADIUS;
dgd.style.borderBottomRightRadius = BORDER_RADIUS;
mhd.style.borderBottomLeftRadius  = BORDER_RADIUS;
mhd.style.borderBottomRightRadius = BORDER_RADIUS;
// Option validators
onchange_dgd_template();
onchange_mhd_svm_kernel();
onchange_mhd_svm_gamma();
onchange_mhd_lr_penalty();
onchange_mhd_dt_max_depth();
onchange_mhd_dt_max_features();
onchange_mhd_ensemble_max_depth();
onchange_mhd_ensemble_max_features();



// ------------------------------ Generate data ------------------------------
function onclick_generate_data() {
    // Get the values
    let cmap       = cmap_input.value;
    let template   = template_input.value;
    let n_samples  = parseInt(n_samples_input.value);
    let noise      = parseFloat(noise_input.value);
    let factor     = parseFloat(factor_input.value);
    let n_classes  = parseInt(n_classes_input.value);
    let class_sep  = parseFloat(class_sep_input.value);

    // Validate the values
    if (isNaN(n_samples)) {alert('Number of samples must be an integer'); return;}
    if (template == 'make_classification') {
        if (isNaN(class_sep)) {alert('Class separation must be a float'); return;}
    }
    else if (template == 'make_moons') {
        if (isNaN(noise)) {alert('Noise must be a float'); return;}
    }
    else if (template == 'make_circles') {
        if (isNaN(noise))  {alert('Noise must be a float');  return;}
        if (isNaN(factor)) {alert('Factor must be a float'); return;}
    }

    // Send the data to the server
    fetch('/api/generate_data', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            template: template,
            n_samples: n_samples,
            noise: noise,
            factor: factor,
            n_classes: n_classes,
            class_sep: class_sep
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.status != 'success') {
            alert(data.error_message);
            return;
        }
        else {
            // Update the plot. Cant do this outside this `then` thingy cause else a race condition occurs and the generated data is not updated
            let randomiser = new Date().getTime(); // Gotta add this so that the image is actually updated
            plot_unclassified.src = `/plot?classifier=none&cmap=${cmap}&randomiser=${randomiser}`;
        }
    });
}



// ------------------------------ Generate plots ------------------------------
function onclick_generate_plots() {
    generate_knn_plot();
    generate_svm_plot();
    generate_lr_plot();
    generate_dt_plot();
    generate_ensemble_plot();
}

function generate_knn_plot() {
    // Get the values
    let cmap            = cmap_input.value;
    let k               = parseInt(mhd_knn_k_input.value);
    let distance_metric = mhd_knn_distance_metric_input.value;

    // Validate the values
    if (isNaN(k)) {alert('k must be an integer'); return;}

    // Update the plot
    let randomiser = new Date().getTime();
    plot_knn.src = `/plot?classifier=knn&cmap=${cmap}&k=${k}&distance_metric=${distance_metric}&randomiser=${randomiser}`;
}

function generate_svm_plot() {
    // Get the values
    let cmap  = cmap_input.value;
    let kernel = mhd_svm_kernel_input.value;
    let c      = parseFloat(mhd_svm_c_input.value);
    let gamma  = mhd_svm_gamma_input.value;
    let degree = parseInt(mhd_svm_degree_input.value);

    // Validate the values
    if (isNaN(c)) {alert('C must be a float'); return;}
    if (kernel == 'poly' || kernel == 'rbf' || kernel == 'sigmoid') {
        if (gamma == 'float') {
            gamma = parseFloat(mhd_svm_gamma_float_input.value);
            if (isNaN(gamma)) {alert('Enter a float for gamma'); return;}
        }
    }
    if (kernel == 'poly') {
        if (isNaN(degree)) {alert('Enter an integer for degree'); return;}
    }

    // Update the plot
    let randomiser = new Date().getTime();
    plot_svm.src = `/plot?classifier=svm&cmap=${cmap}&kernel=${kernel}&c=${c}&gamma=${gamma}&degree=${degree}&randomiser=${randomiser}`;
}

function generate_lr_plot() {
    // Get the values
    let cmap     = cmap_input.value;
    let penalty  = mhd_lr_penalty_input.value;
    let c        = parseFloat(mhd_lr_c_input.value);
    let l1_ratio = parseFloat(mhd_lr_l1_ratio_input.value);

    // Validate the values
    if (isNaN(c)) {alert('C must be a float'); return;}
    if (penalty == 'elasticnet') {
        if (isNaN(l1_ratio)) {alert('Enter a float for l1_ratio'); return;}
    }

    // Update the plot
    let randomiser = new Date().getTime();
    plot_lr.src = `/plot?classifier=logistic_regression&cmap=${cmap}&penalty=${penalty}&c=${c}&l1_ratio=${l1_ratio}&randomiser=${randomiser}`;
}

function generate_dt_plot() {
    // Get the values
    let cmap               = cmap_input.value;
    let criterion          = mhd_dt_criterion_input.value;
    let splitter           = mhd_dt_splitter_input.value;
    let max_depth          = mhd_dt_max_depth_input.value;
    let max_depth_int      = parseInt(mhd_dt_max_depth_int_input.value);
    let max_features       = mhd_dt_max_features_input.value;
    let max_features_int   = parseInt(mhd_dt_max_features_int_input.value);
    let max_features_float = parseFloat(mhd_dt_max_features_float_input.value);

    // Validate the values
    if (max_depth == 'integer') {
        if (isNaN(max_depth_int)) {alert('Enter an integer for max_depth'); return;}
        max_depth = parseInt(max_depth_int);
    }
    if (max_features == 'integer') {
        if (isNaN(max_features_int)) {alert('Enter an integer for max_features'); return;}
        max_features = parseInt(max_features_int);
    }
    if (max_features == 'float') {
        if (isNaN(max_features_float)) {alert('Enter a float for max_features'); return;}
        max_features = parseFloat(max_features_float);
    }

    // Update the plot
    let randomiser = new Date().getTime();
    plot_dt.src = `/plot?classifier=decision_tree&cmap=${cmap}&criterion=${criterion}&splitter=${splitter}&max_depth=${max_depth}&max_features=${max_features}&randomiser=${randomiser}`;
}

function generate_ensemble_plot() {
    // Get the values
    let cmap               = cmap_input.value;
    let n_estimators       = parseInt(mhd_ensemble_n_estimators_input.value);
    let criterion          = mhd_ensemble_criterion_input.value;
    let max_depth          = mhd_ensemble_max_depth_input.value;
    let max_depth_int      = parseInt(mhd_ensemble_max_depth_int_input.value);
    let max_features       = mhd_ensemble_max_features_input.value;
    let max_features_int   = parseInt(mhd_ensemble_max_features_int_input.value);
    let max_features_float = parseFloat(mhd_ensemble_max_features_float_input.value);

    // Validate the values
    if (isNaN(n_estimators)) {alert('Enter an integer for n_estimators'); return;}
    if (max_depth == 'integer') {
        if (isNaN(max_depth_int)) {alert('Enter an integer for max_depth'); return;}
        max_depth = parseInt(max_depth_int);
    }
    if (max_features == 'integer') {
        if (isNaN(max_features_int)) {alert('Enter an integer for max_features'); return;}
        max_features = parseInt(max_features_int);
    }
    if (max_features == 'float') {
        if (isNaN(max_features_float)) {alert('Enter a float for max_features'); return;}
        max_features = parseFloat(max_features_float);
    }

    // Update the plot
    let randomiser = new Date().getTime();
    plot_rf.src = `/plot?classifier=random_forest&cmap=${cmap}&n_estimators=${n_estimators}&criterion=${criterion}&max_depth=${max_depth}&max_features=${max_features}&randomiser=${randomiser}`;
    plot_et.src = `/plot?classifier=extra_trees&cmap=${cmap}&n_estimators=${n_estimators}&criterion=${criterion}&max_depth=${max_depth}&max_features=${max_features}&randomiser=${randomiser}`;
}





// ------------------------------ dgs (data gen summary) event listeners ------------------------------
function change_dgs_color(alpha) {
    dgs.style.backgroundColor = DG_COLOR.replace("{{alpha}}", alpha);
}
function onclick_dgs() {
    if (!is_opened_dgd) {
        change_dgs_color(ACTIVE_ALPHA);
        dgs.style.borderBottomLeftRadius  = "0px";
        dgs.style.borderBottomRightRadius = "0px";
    } else {
        change_dgs_color(INACTIVE_ALPHA);
        dgs.style.borderBottomLeftRadius  = BORDER_RADIUS;
        dgs.style.borderBottomRightRadius = BORDER_RADIUS;
    }
    is_opened_dgd = !is_opened_dgd;
}
function onmouseover_dgs() {change_dgs_color(ACTIVE_ALPHA);}
function onmouseout_dgs() {
    // If the dgd is opened, then don't change the color
    if (!is_opened_dgd) {change_dgs_color(INACTIVE_ALPHA);}
}




// ------------------------------ mhs (models hyperparameters summary) event listeners ------------------------------
function change_mhs_color(alpha) {
    mhs.style.backgroundColor = MH_COLOR.replace("{{alpha}}", alpha);
}
function onclick_mhs() {
    if (!is_opened_mhd) {
        change_mhs_color(ACTIVE_ALPHA);
        mhs.style.borderBottomLeftRadius  = "0px";
        mhs.style.borderBottomRightRadius = "0px";
    } else {
        change_mhs_color(INACTIVE_ALPHA);
        mhs.style.borderBottomLeftRadius  = BORDER_RADIUS;
        mhs.style.borderBottomRightRadius = BORDER_RADIUS;
    }
    is_opened_mhd = !is_opened_mhd;
}
function onmouseover_mhs() {change_mhs_color(ACTIVE_ALPHA);}
function onmouseout_mhs() {
    // If the mhd is opened, then don't change the color
    if (!is_opened_mhd) {change_mhs_color(INACTIVE_ALPHA);}
}




// ------------------------------ dgd input event listeners ------------------------------
function onchange_dgd_template() {
    let template = document.getElementById("dgd-template-input").value;
    console.log(`Changing template to ${template}`)

    if (template == 'make_moons') {
        dg_noise.style.display     = '';
        dg_factor.style.display    = 'none';
        dg_n_classes.style.display = 'none';
        dg_class_sep.style.display = 'none';
    }
    else if (template == 'make_circles') {
        dg_noise.style.display     = '';
        dg_factor.style.display    = '';
        dg_n_classes.style.display = 'none';
        dg_class_sep.style.display = 'none';
    }
    else if (template == 'make_classification') {
        dg_noise.style.display     = 'none';
        dg_factor.style.display    = 'none';
        dg_n_classes.style.display = '';
        dg_class_sep.style.display = '';
    }
    else {
        console.error(`Unknown template: ${template}`);
        alert(`Unknown template: ${template}`);
    }
}




// ------------------------------ mhd input event listeners ------------------------------
function onchange_dgd_template() {
    let template = document.getElementById("dgd-template-input").value;
    console.log(`Changing template to ${template}`)

    if (template == 'make_moons') {
        dg_noise.style.display     = '';
        dg_factor.style.display    = 'none';
        dg_n_classes.style.display = 'none';
        dg_class_sep.style.display = 'none';
    }
    else if (template == 'make_circles') {
        dg_noise.style.display     = '';
        dg_factor.style.display    = '';
        dg_n_classes.style.display = 'none';
        dg_class_sep.style.display = 'none';
    }
    else if (template == 'make_classification') {
        dg_noise.style.display     = 'none';
        dg_factor.style.display    = 'none';
        dg_n_classes.style.display = '';
        dg_class_sep.style.display = '';
    }
    else {
        console.error(`Unknown template: ${template}`);
        alert(`Unknown template: ${template}`);
    }
}

function onchange_mhd_svm_kernel() {
    let kernel = document.getElementById('mhd-svm-kernel-input').value;
    console.log(`[SVM] Changing kernel to ${kernel}`);

    if (kernel == 'linear') {
        mh_svm_degree.style.display = 'none';
        mh_svm_gamma.style.display  = 'none';
    }
    else if (kernel == 'poly') {
        mh_svm_degree.style.display = '';
        mh_svm_gamma.style.display  = '';
    }
    else if (kernel == 'rbf' || kernel == 'sigmoid') {
        mh_svm_degree.style.display = 'none';
        mh_svm_gamma.style.display  = '';
    }
    else {
        console.error(`Unknown kernel: ${kernel}`);
        alert(`Unknown kernel: ${kernel}`);
    }
}

function onchange_mhd_svm_gamma() {
    let value = document.getElementById('mhd-svm-gamma-input').value;
    console.log(`[SVM] Changing gamma to ${value}`);
    
    document.getElementById('mhd-svm-gamma-float').style.display = value=='float' ? 'inline' : 'none';
}

function onchange_mhd_lr_penalty() {
    let penalty = document.getElementById('mhd-lr-penalty-input').value;
    console.log(`[LR] Changing penalty to ${penalty}`);
    
    document.getElementById('mhd-lr-l1-ratio').style.display = penalty=='elasticnet' ? '' : 'none';
}

function onchange_mhd_dt_max_depth() {
    let value = document.getElementById('mhd-dt-max-depth-input').value;
    console.log(`[DT] Changing max_depth to ${value}`);

    document.getElementById('mhd-dt-max-depth-int').style.display = value=='integer' ? 'inline' : 'none';
}

function onchange_mhd_dt_max_features() {
    let value = document.getElementById('mhd-dt-max-features-input').value;
    console.log(`[DT] Changing max_features to ${value}`);

    document.getElementById('mhd-dt-max-features-int').style.display   = value=='integer' ? 'inline' : 'none';
    document.getElementById('mhd-dt-max-features-float').style.display = value=='float'   ? 'inline' : 'none';
}

function onchange_mhd_ensemble_max_depth() {
    let value = document.getElementById('mhd-ensemble-max-depth-input').value;
    console.log(`[Ensemble] Changing max_depth to ${value}`);

    document.getElementById('mhd-ensemble-max-depth-int').style.display = value=='integer' ? 'inline' : 'none';
}

function onchange_mhd_ensemble_max_features() {
    let value = document.getElementById('mhd-ensemble-max-features-input').value;
    console.log(`[Ensemble] Changing max_features to ${value}`);

    document.getElementById('mhd-ensemble-max-features-int').style.display   = value=='integer' ? 'inline' : 'none';
    document.getElementById('mhd-ensemble-max-features-float').style.display = value=='float'   ? 'inline' : 'none';
}