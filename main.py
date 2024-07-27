from flask import Flask, render_template, request, make_response, session
from flask_restful import Api, Resource, request as restful_request
from flask_session import Session

from sklearn.datasets import make_circles, make_moons, make_classification
from sklearn.preprocessing import MinMaxScaler
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

import numpy as np

from base64 import b64encode, b64decode
from io import BytesIO
from typing import Any, Literal
import secrets

import matplotlib.pyplot as plt
import matplotlib.axes
import matplotlib
matplotlib.use('Agg') # https://stackoverflow.com/questions/65068073/error-while-showing-matplotlib-figure-in-flask
plt.style.use('dark_background')
# Change the radius of the points
plt.rcParams['lines.markersize'] = 1
# Turn off the markings in x and y axis
plt.rcParams['xtick.bottom'] = False
plt.rcParams['xtick.labelbottom'] = False
plt.rcParams['ytick.left'] = False
plt.rcParams['ytick.labelleft'] = False
# Set background color to rgb(20,20,20), ie #141414, ie the background of the website
plt.rcParams['figure.facecolor'] = '#141414'



app = Flask(__name__)
api = Api(app)
app.secret_key = secrets.token_urlsafe(16)
app.config['SESSION_TYPE'] = 'cachelib' # https://stackoverflow.com/questions/43304363/simple-server-side-flask-session-variable & https://flask-session.readthedocs.io/en/latest/config.html#SESSION_TYPE
Session(app)


LIMITS = {
    # No model
    'n_samples':               (1,    10_000),
    'noise':                   (0,    1),
    'factor':                  (0,    0.999),
    'n_classes':               (2,    4),
    'class_sep':               (0.1,  5),
    'markersize':              (0.01, 10),
    'alpha':                   (0,    1),
    # SVM
    'degree':                  (1,    10),
    'c':                       (0.01, 10),
    'gamma':                   (0.01, 10),
    # Logistic Regression
    'l1_ratio':                (0,    1),
    # Decision Tree
    'max_depth':               (1,    100),
    'max_features_int':        (1,    100),
    'max_features_float':      (0.01, 1),
    # Random Forest & Extra Trees
    'n_estimators':            (1,    1000),
}

INCLUSIONS = {
    # No model
    'template':           ['make_circles', 'make_moons', 'make_classification'],
    'classifier':         ['none', 'svm', 'knn', 'decision_tree', 'random_forest', 'extra_trees', 'logistic_regression'],
    'cmap':               sorted(plt.colormaps()),
    # KNN
    'distance_metric':    ['chebyshev', 'cosine', 'euclidean', 'haversine', 'manhattan', 'minkowski', 'sqeuclidean'], # seuclidean (Standardised Euclidean) throws an error for some reason
    # SVM
    'kernel':             ['linear', 'poly', 'rbf', 'sigmoid'],
    'gamma':              ['scale', 'auto'],
    # Logistic Regression
    'penalty':            ['l1', 'l2', 'elasticnet', 'none'],
    # Decision Trees
    'criterion':          ['gini', 'entropy', 'log_loss'],
    'splitter':           ['best', 'random'],
    'max_features':       ['none', 'sqrt', 'log2'],
}

TITLES = {
    'none':                '(No classifier)',
    'knn':                 'K-Nearest Neighbours',
    'svm':                 'Support Vector Machine',
    'logistic_regression': 'Logistic Regression',
    'decision_tree':       'Decision Tree',
    'random_forest':       'Random Forest',
    'extra_trees':         'Extra Trees',
}

ERROR_CODES = {
    1:  'template was not provided',
    2:  'n_samples was not provided',
    3:  'noise was not provided',
    4:  'factor was not provided',
    5:  'n_classes was not provided',
    6:  'class_sep was not provided',
    7:  'Invalid template',
    8:  'Invalid value of factor. Must be >=0 and <1',
    9:  'cmap (color map) was not provided',
    10: f'Invalid cmap. Must be one of {INCLUSIONS["cmap"]}',
    11: f'Invalid value of n_samples. Must be >={LIMITS["n_samples"][0]} and <={LIMITS["n_samples"][1]}',
    12: f'Invalid value of markersize. Must be >={LIMITS["markersize"][0]} and <={LIMITS["markersize"][1]}',
    13: 'Markersize was not provided',
    14: 'Classifier was not provided',
    15: f'Invalid classifier. Must be one of {INCLUSIONS["classifier"]}',
    16: 'k was not provided',
    17: 'Invalid value of k. Must be >=1 and <={n_points}',
    18: 'alpha was not provided',
    19: f'Invalid value of alpha. Must be >={LIMITS["alpha"][0]} and <={LIMITS["alpha"][1]}',
    20: 'distance_metric was not provided',
    21: f'Invalid distance_metric. Must be one of {INCLUSIONS["distance_metric"]}',
    22: 'Kernel was not provided',
    23: f'Invalid kernel. Must be one of {INCLUSIONS["kernel"]}',
    24: 'c was not provided',
    25: f'Invalid value of c. Must be >={LIMITS["c"][0]} and <={LIMITS["c"][1]}',
    26: 'gamma was not provided',
    27: f'Invalid value of gamma. Must be one of {INCLUSIONS["gamma"]} or a float >={LIMITS["gamma"][0]} and <={LIMITS["gamma"][1]}',
    28: 'degree was not provided',
    29: f'Invalid value of degree. Must be >={LIMITS["degree"][0]} and <={LIMITS["degree"][1]}',
    30: 'penalty was not provided',
    31: f'Invalid value of penalty. Must be one of {INCLUSIONS["penalty"]}',
    32: 'solver was not provided',
    33: 'Invalid value of solver. Must be one of {valid_solvers}',
    34: 'l1_ratio was not provided',
    35: f'Invalid value of l1_ratio. Must be >={LIMITS["l1_ratio"][0]} and <={LIMITS["l1_ratio"][1]}',
    36: f'Invalid value of n_classes. Must be >={LIMITS["n_classes"][0]} and <={LIMITS["n_classes"][1]}',
    37: 'criterion was not provided',
    38: f'Invalid criterion. Must be one of {INCLUSIONS["criterion"]}',
    39: 'splitter was not provided',
    40: f'Invalid splitter. Must be one of {INCLUSIONS["splitter"]}',
    41: 'max_depth was not provided',
    42: f'Invalid value of max_depth. Must be "none" or an integer X where {LIMITS["max_depth"][0]} <= X <= {LIMITS["max_depth"][1]}',
    43: 'max_features was not provided',
    44: f'Invalid value of max_features. Must be one of {INCLUSIONS["max_features"]}, an integer X where {LIMITS["max_features_int"][0]} <= X <= {LIMITS["max_features_int"][1]}, or a float Y where {LIMITS["max_features_float"][0]} <= Y <= {LIMITS["max_features_float"][1]}',
    45: 'n_estimators was not provided',
    46: f'Invalid value of n_estimators. Must be >={LIMITS["n_estimators"][0]} and <={LIMITS["n_estimators"][1]}',
    47: f'Invalid value of class_sep. Must be >={LIMITS["class_sep"][0]} and <={LIMITS["class_sep"][1]}',
    48: 'n_samples is not a valid integer',
    49: f'Invalid value of noise. Must be >={LIMITS["noise"][0]} and <={LIMITS["noise"][1]}',
    50: f'Invalid value of class_sep. Must be >={LIMITS["class_sep"][0]} and <={LIMITS["class_sep"][1]}',
}

def ERROR(error_code:int, **kwargs) -> tuple[dict[str, Any], Literal[400]]:
    return {
        'status':       'error',
        'error_code':    error_code,
        'error_message': ERROR_CODES[error_code].format(**kwargs)
    }, 400







class ApiGenerateDataset(Resource):
    def func(self):
        # session.clear()
        # Get the parameters given to us via the POST/GET request
        arguments = ['template', 'n_samples', 'factor', 'noise', 'n_classes', 'class_sep']
        args = {}
        for i in arguments:
            args[i] = request.args.get(i,
                      restful_request.get_json().get(i))
        
        # Check if the template was provided and if it is valid
        if args['template'] is None:                          return ERROR(1)
        if args['template'] not in INCLUSIONS['template']:    return ERROR(7)
        # Check if n_samples was provided
        if args['n_samples'] is None:    return ERROR(2)
        # Check if n_samples is a valid integer
        try:                  int(args['n_samples'])
        except ValueError:    return ERROR(48)
        # Check if n_samples is in the valid range
        if not ( LIMITS['n_samples'][0] <= int(args['n_samples']) <= LIMITS['n_samples'][1] ):    return ERROR(11)
        
        # If the template is make_moons, check if noise was provided
        if args['template'] == 'make_moons':
            if args['noise'] is None:    return ERROR(3)
        # If the template is make_circles, check if factor & noise were provided
        elif args['template'] == 'make_circles':
            if args['noise']  is None:    return ERROR(3)
            if args['factor'] is None:    return ERROR(4)
        # If the template is classification, check if n_classes and class_sep were provided, and if n_classes is valid
        elif args['template'] == 'make_classification':
            if args['n_classes'] is None:                                                              return ERROR(5)
            if args['class_sep'] is None:                                                              return ERROR(6)
            if not ( LIMITS['n_classes'][0] <= int(args['n_classes']) <= LIMITS['n_classes'][1] ):     return ERROR(36)
            if not ( LIMITS['class_sep'][0] <= float(args['class_sep']) <= LIMITS['class_sep'][1] ):   return ERROR(47)
        
        # If noise is given, check if it is valid
        if args['noise'] is not None and not ( LIMITS['noise'][0] <= float(args['noise']) <= LIMITS['noise'][1] ):    return ERROR(49)
        # If class_sep is given, check if it is valid
        if args['class_sep'] is not None and not ( LIMITS['class_sep'][0] <= float(args['class_sep']) <= LIMITS['class_sep'][1] ):    return ERROR(50)
        # If the value of factor is given, check if it is valid
        if args['factor'] is not None and not ( LIMITS['factor'][0] <= float(args['factor']) <= LIMITS['factor'][1] ):    return ERROR(8)
        # If the value of class_sep is given, check if it is valid
        if args['class_sep'] is not None and not ( LIMITS['class_sep'][0] <= float(args['class_sep']) <= LIMITS['class_sep'][1] ):    return ERROR(50)


        # Generate the data (ie label data and attribute data)
        kwargs = {}
        kwargs['n_samples'] = int(args['n_samples'])
        if args['template'] == 'make_circles':
            data_gen_func    = make_circles
            kwargs['noise']  = float(args['noise'])
            kwargs['factor'] = float(args['factor'])
        elif args['template'] == 'make_moons':
            data_gen_func   = make_moons
            kwargs['noise'] = float(args['noise'])
        elif args['template'] == 'make_classification':
            data_gen_func = make_classification
            kwargs['n_classes']     = int(args['n_classes'])
            kwargs['class_sep']     = float(args['class_sep'])
            kwargs['n_features']    = 2
            kwargs['n_informative'] = 2
            kwargs['n_redundant']   = 0
            kwargs['n_clusters_per_class'] = 1
        attrs_data, label_data = data_gen_func(**kwargs)
        # Scale the data
        attrs_data = MinMaxScaler().fit_transform(attrs_data)

        # Save the attributes and label data in cookies
        # response = make_response({'status': 'Success'})
        # expiration_time = datetime.now() + timedelta(days=365)

        attrs_data_base64 = b64encode(attrs_data).decode('utf-8')
        label_data_base64 = b64encode(label_data).decode('utf-8')
        session['attrs_data'] = attrs_data_base64
        session['label_data'] = label_data_base64
        # response.set_cookie('label_data', label_data_base64, expires=expiration_time)
        # response.set_cookie('attrs_data', attrs_data_base64, expires=expiration_time)

        return {'status': 'success'}

    def get(self):  return self.func()
    def post(self): return self.func()
api.add_resource(ApiGenerateDataset, '/api/generate_data')





@app.route('/')
def node_root():
    return render_template('index.html')


@app.route('/plot')
def node_plot():
    args = {}
    needed_args  = ['classifier', 'cmap']                        # No model
    needed_args += ['k', 'distance_metric']                      # KNN
    needed_args += ['kernel', 'c', 'gamma', 'degree']            # SVM
    needed_args += ['penalty', 'c', 'l1_ratio']                  # Logistic Regression
    needed_args += ['criterion', 'splitter', 'max_depth', 'max_features']     # Decision Trees
    needed_args += ['n_estimators', 'criterion', 'max_depth', 'max_features'] # Random Forest & Extra Trees
    for i in needed_args:
        args[i] = request.args.get(i,
                  request.form.get(i))

    # Check if classifier was provided and if it is valid
    if args['classifier'] is None:                            return ERROR(14)
    if args['classifier'] not in INCLUSIONS['classifier']:    return ERROR(15)
    # Check if cmap was provided and if it is valid
    if args['cmap'] is None:                      return ERROR(9)
    if args['cmap'] not in INCLUSIONS['cmap']:    return ERROR(10)

    # Get the attributes and label data from the session
    decoded_attrs_data = b64decode(session['attrs_data'])
    decoded_label_data = b64decode(session['label_data'])
    attrs_data = np.frombuffer(decoded_attrs_data, dtype=np.float64).reshape(-1, 2)
    label_data = np.frombuffer(decoded_label_data, dtype=np.int64)
    

    # If the classifier is KNN
    if args['classifier'] == 'knn':
        n_points = len(attrs_data)
        # Check if k was provided and if it is valid
        if args['k'] is None:                          return ERROR(16)
        if not ( 1 <= int(args['k']) <= n_points ):    return ERROR(17, n_points=n_points)
        # Check if distance_metric was provided and if it is valid
        if args['distance_metric'] is None:                                 return ERROR(20)
        if args['distance_metric'] not in INCLUSIONS['distance_metric']:    return ERROR(21)
        # Create the classifier
        clf = KNeighborsClassifier( n_neighbors=int(args['k']) , metric=args['distance_metric'] )


    # If the classifier is SVM
    elif args['classifier'] == 'svm':
        # Check if kernel was provided and if it is valid
        if args['kernel'] is None:                        return ERROR(22)
        if args['kernel'] not in INCLUSIONS['kernel']:    return ERROR(23)
        # Check if c was provided and if it is valid
        if args['c'] is None:                                               return ERROR(24)
        if not ( LIMITS['c'][0] <= float(args['c']) <= LIMITS['c'][1] ):    return ERROR(25)
        # If the kernel is 'poly', check if degree was provided and if it is valid
        if args['kernel'] == 'poly':
            if args['degree'] is None:                                                       return ERROR(28)
            if not ( LIMITS['degree'][0] <= int(args['degree']) <= LIMITS['degree'][1] ):    return ERROR(29)
        # If the kernel is 'rbf', 'poly', or 'sigmoid', check if gamma was provided and if it is valid
        if args['kernel'] in ['rbf', 'poly', 'sigmoid']:
            if args['gamma'] is None:    return ERROR(26)
            # If gamma is a float, check if it is valid
            try:
                if not ( LIMITS['gamma'][0] <= float(args['gamma']) <= LIMITS['gamma'][1] ):    return ERROR(27)
            # If gamma is not a float, check if it is valid
            except ValueError:
                if args['gamma'] not in INCLUSIONS['gamma']:    return ERROR(27)
        # Specify the arguments
        kwargs = {'kernel': args['kernel'], 'C': float(args['c'])}
        if args['kernel'] == 'poly':    kwargs['degree'] = int(args['degree'])
        if args['kernel'] in ['rbf', 'poly', 'sigmoid']:
            if args['gamma'] in INCLUSIONS['gamma']:    kwargs['gamma'] = args['gamma']
            else:                                       kwargs['gamma'] = float(args['gamma'])
        # Create the classifier
        clf = SVC(**kwargs)


    # If the classifier is logistic regression
    elif args['classifier'] == 'logistic_regression':
        # Check if penalty was provided and if it is valid
        if args['penalty'] is None:                         return ERROR(30)
        if args['penalty'] not in INCLUSIONS['penalty']:    return ERROR(31)
        # Check if c was provided and if it is valid
        if args['c'] is None:                                               return ERROR(24)
        if not ( LIMITS['c'][0] <= float(args['c']) <= LIMITS['c'][1] ):    return ERROR(25)
        # If the penalty is 'elasticnet', check if l1_ratio was provided and if it is valid
        if args['penalty'] == 'elasticnet':
            if args['l1_ratio'] is None:                                                             return ERROR(34)
            if not ( LIMITS['l1_ratio'][0] <= float(args['l1_ratio']) <= LIMITS['l1_ratio'][1] ):    return ERROR(35)
        # If the penalty is 'none'
        if args['penalty'] == 'none':    penalty = None
        else:                            penalty = args['penalty']
        # If the penalty is 'elasticnet', set the l1_ratio to the given float. Else, set it to None
        l1_ratio = float(args['l1_ratio']) if args['penalty'] == 'elasticnet' else None
        # Create the classifier
        clf = LogisticRegression( penalty=penalty , C=float(args['c']) , l1_ratio=l1_ratio , solver='saga' )


    # If the classifier is decision tree
    elif args['classifier'] == 'decision_tree':
        # Check if criterion was provided and if it is valid
        if args['criterion'] is None:                           return ERROR(37)
        if args['criterion'] not in INCLUSIONS['criterion']:    return ERROR(38)
        # Check if splitter was provided and if it is valid
        if args['splitter'] is None:                          return ERROR(39)
        if args['splitter'] not in INCLUSIONS['splitter']:    return ERROR(40)
        # Check if max_depth was provided
        if args['max_depth'] is None:      return ERROR(41)
        if args['max_depth'] == 'none':    max_depth = None
        # If it is not 'none', check if it is an integer
        elif args['max_depth'].isdigit():
            max_depth = int(args['max_depth'])
            if not ( LIMITS['max_depth'][0] <= max_depth <= LIMITS['max_depth'][1] ):    return ERROR(42)
        # Else, it is invalid
        else:    return ERROR(42)
        # Check if max_features was provided
        if args['max_features'] is None:    return ERROR(43)
        if args['max_features'] in INCLUSIONS['max_features']:
            max_features = args['max_features'] if args['max_features'] != 'none' else None
        # If it is not one of the included words, check if it is a float
        elif '.' in args['max_features'] and args['max_features'].replace('.', '').isdigit():
            max_features = float(args['max_features'])
            if not ( LIMITS['max_features_float'][0] <= max_features <= LIMITS['max_features_float'][1] ):    return ERROR(44)
        # If it is not a float, check if it is an integer
        elif args['max_features'].isdigit():
            max_features = int(args['max_features'])
            if not ( LIMITS['max_features_int'][0] <= max_features <= LIMITS['max_features_int'][1] ):    return ERROR(44)
        # Else, it is invalid
        else:    return ERROR(44)
        # Create the classifier
        clf = DecisionTreeClassifier( criterion=args['criterion'], splitter=args['splitter'], max_depth=max_depth, max_features=max_features )
    

    # If the classifier is random forest or extra trees (since they have the same hyperparameters)
    elif args['classifier'] in ['random_forest', 'extra_trees']:
        # Check if n_estimators was provided and if it is valid
        if args['n_estimators'] is None:                                                                   return ERROR(45)
        if not ( LIMITS['n_estimators'][0] <= int(args['n_estimators']) <= LIMITS['n_estimators'][1] ):    return ERROR(46)
        # Check if criterion was provided and if it is valid
        if args['criterion'] is None:                           return ERROR(37)
        if args['criterion'] not in INCLUSIONS['criterion']:    return ERROR(38)
        # Check if max_depth was provided
        if args['max_depth'] is None:      return ERROR(41)
        if args['max_depth'] == 'none':    max_depth = None
        # If it is not 'none', check if it is an integer
        elif args['max_depth'].isdigit():
            max_depth = int(args['max_depth'])
            if not ( LIMITS['max_depth'][0] <= max_depth <= LIMITS['max_depth'][1] ):    return ERROR(42)
        # Else, it is invalid
        else:    return ERROR(42)
        # Check if max_features was provided
        if args['max_features'] is None:    return ERROR(43)
        if args['max_features'] in INCLUSIONS['max_features']:
            max_features = args['max_features'] if args['max_features'] != 'none' else None
        # If it is not one of the included words, check if it is a float
        elif '.' in args['max_features'] and args['max_features'].replace('.', '').isdigit():
            max_features = float(args['max_features'])
            if not ( LIMITS['max_features_float'][0] <= max_features <= LIMITS['max_features_float'][1] ):    return ERROR(44)
        # If it is not a float, check if it is an integer
        elif args['max_features'].isdigit():
            max_features = int(args['max_features'])
            if not ( LIMITS['max_features_int'][0] <= max_features <= LIMITS['max_features_int'][1] ):    return ERROR(44)
        # Else, it is invalid
        else:    return ERROR(44)
        # Create the classifier
        if args['classifier'] == 'random_forest':    clf = RandomForestClassifier( n_estimators=int(args['n_estimators']) , criterion=args['criterion'] , max_depth=max_depth , max_features=max_features )
        else:                                        clf = ExtraTreesClassifier  ( n_estimators=int(args['n_estimators']) , criterion=args['criterion'] , max_depth=max_depth , max_features=max_features )
    

    # Plot the data
    plt.rcParams['image.cmap'] = args['cmap']
    fig, ax = plt.subplots(1,1)
    ax: matplotlib.axes.Axes
    ax.set_title(TITLES[args['classifier']])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.scatter(attrs_data[:,0], attrs_data[:,1], c=label_data)

    # If a classifier is given, plot the decision boundary
    if args['classifier'] != 'none':
        clf.fit(attrs_data, label_data)
        DecisionBoundaryDisplay.from_estimator(clf, attrs_data, ax=ax, alpha=0.4, cmap=args['cmap'])

    # Save the plot as a byte stream
    output = BytesIO()
    matplotlib.backends.backend_agg.FigureCanvasAgg(fig).print_png(output)
    plt.close(fig) # Avoiding memory leaks
    response = make_response(output.getvalue())
    response.content_type = 'image/png'
    response.mimetype     = 'image/png'
    return response 
    # return Response(output.getvalue(), mimetype='image/png')




if __name__ == '__main__':
    app.run(debug=True, port=8000, host='0.0.0.0', threaded=True)