#
# C O N S T A N T S
#
# Various constants

# This is just the text that will be placed on the visible treatment plan
SPRAYER_NAME = ""

MSG_NOT_CONNECTED = "Not connected"
MSG_LINES_NOT_SPECIFIED = "The A and B lines for the odometer must be on the command line or in the INI file"
MSG_ODOMETER_CONNECTION = "Could not connect to odometer"

MSG_NO_PROBLEM_FOUND = "No problem found"

# Names for the attributes
# Some of these are a bit redundant and are there for backwards compatibility

NAME_AREA                   = "area"
NAME_TYPE                   = "type"
NAME_LOCATION               = "location"
NAME_CENTER                 = "center"
NAME_CONTOUR                = "contour"
NAME_SHAPE_INDEX            = "shape_index"
NAME_RATIO                  = "lw_ratio"
NAME_IMAGE                  = "image"
NAME_REASON                 = "reason"
NAME_SIZE_RATIO             = "size_ratio"
NAME_BLUE                   = "blue"
NAME_DISTANCE               = "distance"
NAME_DISTANCE_NORMALIZED    = "normalized_distance"
NAME_NAME                   = "name"
NAME_NUMBER                 = "number"
NAME_HEIGHT                 = "height"
NAME_NEIGHBOR_COUNT         = "neighbors"
NAME_HUE                    = "hue"
NAME_HUE_MEAN               = "hue_mean"
NAME_HUE_STDDEV             = "hue_stddev"
NAME_SATURATION             = "saturation_mean"
NAME_SATURATION_STDEV       = "saturation_stddev"
NAME_I_YIQ                  = "in_phase"
NAME_I_MEAN                 = "in_phase_mean"
NAME_I_STDDEV               = "in_phase_stddev"
NAME_Q_MEAN                 = "quadrature_mean"
NAME_Q_STDDEV               = "quadrature_stddev"
NAME_BLUE_DIFFERENCE        = "cb_mean"
NAME_BLUE_DIFFERENCE_MEAN   = "cb_mean"
NAME_BLUE_DIFFERENCE_STDEV  = "cb_stddev"
NAME_RED_DIFFERENCE_MEAN    = "cr_mean"
NAME_RED_DIFFERENCE_STDEV   = "cr_stddev"
NAME_COMPACTNESS            = "compactness"
NAME_ELONGATION             = "elongation"
NAME_ECCENTRICITY           = "eccentricity"
NAME_ROUNDNESS              = "roundness"
NAME_CONVEXITY              = "convexity"
NAME_SOLIDITY               = "solidity"

#
# X M P P  R O O M S  A N D  J I D S
#
CONFERENCE_DNS              = "conference.weeds.com"
ROOM_ODOMETRY               = "odometry" + "@" + CONFERENCE_DNS
ROOM_TREATMENT              = "treatment" + "@" + CONFERENCE_DNS


# TODO: There are two Jetsons in the system. We need a scheme for controller-1@weeds.local and controller-2.weeds.local

JID_ODOMETRY                = "rio@weeds.com"
JID_RIO                     = "rio@weeds.com"
JID_JETSON                  = "controller@weeds.com"
JID_JETSON_1                = "jetson1@weeds.com"
JID_JETSON_2                = "jetson2@weeds.com"
JID_CONSOLE                 = "console@weeds.com"


DEFAULT_PASSWORD            = "greydog"
#DEFAULT_PASSWORD           = "weeds"

NICK_ODOMETRY               = "rio"
NICK_TREATMENT              = "treatment"
NICK_JETSON                 = "controller"
NICK_JETSON_1               = "jetson1"
NICK_JETSON_2               = "jetson2"
NICK_CONSOLE                = "console"
MSG_TYPE_NORMAL             = "normal"
MSG_TYPE_GROUPCHAT          = "groupchat"

# The timeout period for the processing
PROCESS_TIMEOUT             = 10.0

# Temporary
NAME_CROP_SCORE = "score"

# A shortcut used for the command line
NAME_ALL = "all"
NAME_NONE = "none"

#
# R I O
#
RIO_TASK_NAME = "readEncoder"

#
# I M A G E S
#
# Image acquisition strategy
STRATEGY_ASYNC = 0
STRATEGY_SYNC = 1

IMAGE_QUEUE_LEN = 500
# Probably way too long 5000 ms for camera to acquire image
TIMEOUT_CAMERA = 5000

KEYWORD_DIRECTORY = "directory"
KEYWORD_IP = "ip"

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
REASON_SVM = 8
REASON_TOO_CLOSE = 9
REASONS = ["Unknown", "At Edge", "Size", "Logistic", "KNN", "Decision Tree", "Random Forest", "Gradient", "SVM", "Too Close"]

# TYPES of vegetation
TYPE_DESIRED = 0
TYPE_UNDESIRED = 1
TYPE_UNTREATED = 2
TYPE_IGNORED = 3
TYPE_UNKNOWN = 4
TYPES = ["Desired", "Undesired", "Untreated", "Ignored", "Unknown"]

# Items contained in performance csv for analysis

PERF_ANGLES     = "angles"
PERF_ACQUIRE    = "acquire"
PERF_CLASSIFY   = "classify"
PERF_CONTOURS   = "contours"
PERF_INDEX      = "index"
PERF_LW_RATIO   = "LWRatio"
PERF_OVERLAP    = "overlap"
PERF_REGRESSION = "regression"
PERF_SHAPES_IDX = "shapes_idx"
PERF_TREATMENT  = "treatment"
PERF_COLORS     = "colors"
PERF_HSI        = "hsi"
PERF_HSV        = "hsv"
PERF_YIQ        = "yiq"
PERF_YCC        = "ycc"
PERF_MEAN       = "mean"
PERF_STDDEV     = "stddev"
PERF_COMPACTNESS= "compactness"
PERF_SHAPES     = "shapes"
PERF_UNDISTORT  = "undistort"

PERF_TITLE_ACTIVITY = "activity"
PERF_TITLE_MILLISECONDS = "milliseconds"

COLOR_TREATMENT_GRID = (0,255,0)
COLOR_TREATMENT_WEED = (0,0,255)
COLOR_TREATMENT_SPRAYER_ON = (0,0,255)
COLOR_TREATMENT_SPRAYER_OFF = (0,0,255)
SIZE_TREATMENT_LINE = 4
SIZE_CONTOUR_LINE = 5

# Properties
PROPERTY_FILENAME = "options.ini"
PROPERTY_PIXELS_PER_MM = "PIXELS-PER-MM"
PROPERTY_SECTION_CAMERA = "CAMERA"
PROPERTY_CAMERA_IP = "IP" #"169.254.212.41"

# O D O M E T E R
PROPERTY_SECTION_ODOMETER = "ODOMETER"
PROPERTY_LINE_A = "LINE-A"
PROPERTY_LINE_B = "LINE-B"
PROPERTY_LINE_Z = "LINE-Z"
PROPERTY_PFI_A = "PFI-A"
PROPERTY_PFI_B = "PFI-B"
PROPERTY_PFI_Z = "PFI-Z"
PROPERTY_PPR = "PULSES"
PROPERTY_WHEEL_CIRCUMFERENCE = "WHEEL-SIZE"
PROPERTY_ENCODER_TYPE = "TYPE"
PROPERTY_DEBOUNCE = "DEBOUNCE"
PROPERTY_ANNOUNCEMENTS = "ANNOUNCEMENTS"
#
# X M P P  R O O M S  A N D  J I D S
#
PROPERTY_SECTION_XMPP                = "XMPP"
PROPERTY_DOMAIN_DNS                  = "DOMAIN"
PROPERTY_CONFERENCE_DNS              = "CONFERENCE_DNS"
PROPERTY_ROOM_ODOMETRY               = "ROOM_ODOMETRY"
PROPERTY_ROOM_TREATMENT              = "ROOM_TREATMENT"
PROPERTY_ROOM_SYSTEM                 = "ROOM_SYSTEM"

PROPERTY_JID_ODOMETRY                = "JID_ODOMETRY"
PROPERTY_JID_RIO                     = "JID_RIO"
PROPERTY_JID_JETSON                  = "JID_JETSON"
PROPERTY_JID_JETSON_1                = "JID_JETSON1"
PROPERTY_JID_JETSON_2                = "JID_JETSON2"
PROPERTY_JID_CONSOLE                 = "JID_CONSOLE"


PROPERTY_DEFAULT_PASSWORD            = "DEFAULT_PASSWORD"

PROPERTY_NICK_ODOMETRY               = "NICK_ODOMETRY"
PROPERTY_NICK_TREATMENT              = "NICK_TREATMENT"
PROPERTY_NICK_JETSON                 = "NICK_JETSON"
PROPERTY_NICK_JETSON_1               = "NICK_JETSON1"
PROPERTY_NICK_JETSON_2               = "NICK_JETSON2"
PROPERTY_NICK_CONSOLE                = "NICK_CONSOLE"
PROPERTY_MSG_TYPE_NORMAL             = "MSG_TYPE_NORMAL"
PROPERTY_MSG_TYPE_GROUPCHAT          = "MSG_TYPE_GROUPCHAT"

#
# XML
#
XML_ROOT = "root"
XML_DISTANCE = "distance"

