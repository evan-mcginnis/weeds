#
# C O N S T A N T S
#
# Various constants

#
SPRAYER_NAME = "SPRAYER"

# Names for the attributes
NAME_AREA = "area"
NAME_TYPE = "type"
NAME_LOCATION = "location"
NAME_CENTER = "center"
NAME_CONTOUR = "contour"
NAME_SHAPE_INDEX = "shape_index"
NAME_RATIO = "lw_ratio"
NAME_IMAGE = "image"
NAME_REASON = "reason"
NAME_SIZE_RATIO = "size_ratio"
NAME_BLUE = "blue"
NAME_DISTANCE = "distance"
NAME_DISTANCE_NORMALIZED = "normalized_distance"
NAME_NAME = "name"
NAME_NUMBER = "number"
NAME_HEIGHT = "height"
NAME_NEIGHBOR_COUNT = "neighbors"
NAME_HUE = "hue"
NAME_HUE_MEAN = "hue_mean"
NAME_SATURATION = "saturation_mean"
NAME_I_YIQ = "in_phase"
NAME_BLUE_DIFFERENCE = "cb_mean"



# Temporary
NAME_CROP_SCORE = "score"

# A shortcut used for the command line
NAME_ALL = "all"
NAME_NONE = "none"

names = [NAME_AREA, NAME_TYPE, NAME_LOCATION, NAME_CENTER, NAME_CONTOUR, NAME_SHAPE_INDEX, NAME_RATIO, NAME_IMAGE, NAME_REASON, NAME_HUE_MEAN, NAME_SATURATION]

# REASONS why things were scored the way they were
REASON_UNKNOWN = 0
REASON_AT_EDGE = 1
REASON_SIZE_RATIO = 2
REASON_LOGISTIC_REGRESSION = 3
REASON_KNN = 4
REASON_DECISION_TREE = 5
REASON_RANDOM_FOREST = 6
REASON_GRADIENT = 7
REASONS = ["Unknown", "At Edge", "Size", "Logistic", "KNN", "Decision Tree", "Random Forest", "Gradient"]

# TYPES of vegetation
TYPE_DESIRED = 0
TYPE_UNDESIRED = 1
TYPE_UNTREATED = 2
TYPE_IGNORED = 3
TYPE_UNKNOWN = 4
TYPES = ["Desired", "Undesired", "Untreated", "Ignored", "Unknown"]

# Items contained in performance csv for analysis

PERF_ANGLES = "angles"
PERF_ACQUIRE = "acquire"
PERF_CLASSIFY = "classify"
PERF_CONTOURS = "contours"
PERF_INDEX = "index"
PERF_LW_RATIO = "LWRatio"
PERF_OVERLAP = "overlap"
PERF_REGRESSION = "regression"
PERF_SHAPES = "shapes"
PERF_TREATMENT = "treatment"
PERF_COLORS = "colors"
PERF_HSI = "hsi"
PERF_HSV = "hsv"
PERF_YIQ = "yiq"
PERF_YCC = "ycc"
PERF_MEAN = "mean"
PERF_STDDEV = "stddev"

PERF_TITLE_ACTIVITY = "activity"
PERF_TITLE_MILLISECONDS = "milliseconds"

COLOR_TREATMENT_GRID = (0,255,0)
COLOR_TREATMENT_WEED = (0,0,255)

# Properties
PROPERTY_PIXELS_PER_MM = "PIXELS-PER-MM"
