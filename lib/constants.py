#
# C O N S T A N T S
#
from enum import Enum

# UI
UI_STATUS_OK = "OK"
UI_STATUS_NOT_OK = "Not OK"
UI_STATUS_UNKNOWN = "Unknown"
UI_OPERATION_IMAGING = "Imaging only"
UI_OPERATION_WEEDING = "Weeding"
UI_OPERATION_NONE = "No operation in progress"
UI_CONFIRM_IMAGING_ALL_OK = "Begin Imaging?"
UI_CONFIRM_IMAGING_WITH_ERRORS = "Not all components are operational. Do you want to begin imaging?"

# File names
PREFIX_CAMERA_CONFIGURATION = "camera-configuration-"
EXTENSION_CAMERA_SETTINGS = ".pfs"
EXTENSION_IMAGE = ".jpg"
EXTENSION_NPY = ".npy"
EXTENSION_NPZ = ".npz"
EXTENSION_META = ".meta"
EXTENSION_CSV = ".csv"

FILENAME_RAW = "raw"
FILENAME_FINISHED = "img"

# This is just the text that will be placed on the visible treatment plan
SPRAYER_NAME = ""

THREAD_NAME_ODOMETRY = "odometry"
THREAD_NAME_SYSTEM = "system"
THREAD_NAME_TREATMENT = "treatment"
THREAD_NAME_ACQUIRE = "acquire"
THREAD_NAME_SERVICE = "service"
THREAD_NAME_HOUSEKEEPING = "housekeeping"
THREAD_NAME_IMU = "imu"
THREAD_NAME_DEPTH = "depth"
# These next two are temporary -- not needed when we move the depth cameras to the jetson
THREAD_NAME_DEPTH_LEFT = "depth_left"
THREAD_NAME_DEPTH_RIGHT = "depth_right"
THREAD_NAME_POSITION = "position"
THREAD_NAME_SUPERVISOR = "supervisor"
THREAD_NAME_ENRICH = "enrich"

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
NAME_DIST_TO_LEADING_EDGE   = "leading"
#
# X M P P  R O O M S  A N D  J I D S
#
CONFERENCE_DNS              = "conference.weeds.com"
ROOM_ODOMETRY               = "odometry" + "@" + CONFERENCE_DNS
ROOM_TREATMENT              = "treatment" + "@" + CONFERENCE_DNS
ROOM_SYSTEM                 = "system" + "@" + CONFERENCE_DNS


# TODO: There are two Jetsons in the system. We need a scheme for controller-1@weeds.local and controller-2.weeds.local

JID_ODOMETRY                = "rio@weeds.com"
JID_RIO                     = "rio@weeds.com"
JID_JETSON                  = "controller@weeds.com"
JID_JETSON_1                = "jetson1@weeds.com"
JID_JETSON_2                = "jetson2@weeds.com"
JID_CONSOLE                 = "console@weeds.com"


DEFAULT_PASSWORD            = "greydog"
#DEFAULT_PASSWORD           = "weeds"

NICK_CONTROLER              = "controller"
NICK_ODOMETRY               = "rio"
NICK_TREATMENT              = "treatment"
NICK_JETSON                 = "jetson"
NICK_JETSON_1               = "left"
NICK_JETSON_2               = "right"
NICK_CONSOLE                = "console"
MSG_TYPE_NORMAL             = "normal"
MSG_TYPE_GROUPCHAT          = "groupchat"

# The timeout period for the processing
PROCESS_TIMEOUT             = 10000.0
PROCESS_TIMEOUT_LONG        = 10000.0
PROCESS_TIMEOUT_SHORT       = 4.0
KEYWORD_TIMEOUT             = "TIMEOUT"

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

# The number of images to keep
IMAGE_QUEUE_LEN = 50
DEPTH_QUEUE_LEN =5
# Probably way too long 5000 ms for camera to acquire image
TIMEOUT_CAMERA = 5000

KEYWORD_DIRECTORY = "directory"
KEYWORD_IP = "ip"
KEYWORD_FILE_GYRO = "gyro"
KEYWORD_FILE_ACCELERATION = "acceleration"
KEYWORD_SERIAL = "serial"
KEYWORD_CONFIGURATION = "config"

PARAM_FILE_GYRO = "gyro.csv"
PARAM_FILE_ACCELERATION = "acceleration.csv"

#
# D E P T H
#
DEPTH_MAX_HORIZONTAL = 1280
DEPTH_MAX_VERTICAL = 720
DEPTH_MAX_FRAMES = 6

#
# I M U
#
# The files that will hold the IMU information for the weeding session

FILE_GYRO = "gyro.csv"
FILE_ACCELERATION = "acceleration.csv"
FILE_DEPTH = "depth-{:03d}.npy"

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
PERF_DISTANCE   = "distance"
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
PROPERTY_DNS_SERVER = "DNS"
PROPERTY_LOGGING_FILENAME = "logging.ini"
PROPERTY_DISTANCE_TIER = "DISTANCE-TIERS"
PROPERTY_DISTANCE_EMITTERS = "DISTANCE-EMITTERS"
PROPERTY_FILENAME = "options.ini"
PROPERTY_FILENAME_FINISHED = "FINISHED"
PROPERTY_PIXELS_PER_MM = "PIXELS-PER-MM"
PROPERTY_SECTION_CAMERA = "CAMERA"
PROPERTY_SECTION_EMITTER = "EMITTERS"
PROPERTY_CAMERA_IP = "IP" #"169.254.212.41"
PROPERTY_IMAGE_WIDTH = "IMAGE_WIDTH"
PROPERTY_IMAGE_HEIGHT = "IMAGE_HEIGHT"
PROPERTY_OFFSET_X = "EMITTER-X-OFFSET"
PROPERTY_OFFSET_Y = "EMITTER-Y-OFFSET"
PROPERTY_OVERLAP_FACTOR = "OVERLAP_FACTOR"
PROPERTY_RESOLUTION = "RESOLUTION"
PROPERTY_OUTPUT = "OUTPUT"
PROPERTY_SERIAL = "SERIAL"
PROPERTY_LOCATION = "LOCATION"
# Temporary until the depth cameras are moved to the NVIDIA systems
PROPERTY_SERIAL_LEFT = "SERIAL-LEFT"
PROPERTY_SERIAL_RIGHT = "SERIAL-RIGHT"

# D E P T H
PROPERTY_SECTION_DEPTH = "DEPTH"
PROPERTY_AGL_UP = "AGL_UP"
PROPERTY_AGL_DOWN = "AGL_DOWN"
PROPERTY_AGL_STOP = "AGL_STOP"

# R I O
PROPERTY_SECTION_RIO = "RIO"
PROPERTY_RIGHT = "RIGHT-9403"
PROPERTY_LEFT = "LEFT-9403"
PROPERTY_CARD_ODOMETER = "NAME-9411"

# O D O M E T E R
PROPERTY_SECTION_ODOMETER = "ODOMETER"
PROPERTY_FORWARD = "FORWARD"
PROPERTY_LINE_A = "LINE-A"
PROPERTY_LINE_B = "LINE-B"
PROPERTY_LINE_Z = "LINE-Z"
PROPERTY_PFI_A = "PFI-A"
PROPERTY_PFI_B = "PFI-B"
PROPERTY_PFI_Z = "PFI-Z"
PROPERTY_PPR = "PULSES"
PROPERTY_COUNTER = "COUNTER"
PROPERTY_WHEEL_CIRCUMFERENCE = "WHEEL-SIZE"
PROPERTY_ENCODER_TYPE = "TYPE"
PROPERTY_DEBOUNCE = "DEBOUNCE"
PROPERTY_ANNOUNCEMENTS = "ANNOUNCEMENTS"

# Things that will be passed in keyword arguments
KEYWORD_LINE_A = "LINE_A"
KEYWORD_LINE_B = "LINE_B"
KEYWORD_WHEEL_CIRCUMFERENCE = "WHEEL_SIZE"
KEYWORD_PPR = "PULSES"
KEYWORD_SPEED = "SPEED"
KEYWORD_COUNTER = "COUNTER"
KEYWORD_FORWARD = "FORWARD"
KEYWORD_GSD = "GSD"
KEYWORD_CALLBACK = "CALLBACK"
KEYWORD_TYPE = "TYPE"
KEYWORD_EXPOSURE = "EXPOSURE"
KEYWORD_LAT = "LATITUDE"
KEYWORD_LONG = "LONGITUDE"
KEYWORD_PULSES = "PULSES"
KEYWORD_SOFTWARE = "SOFTWARE"
KEYWORD_COPYRIGHT = "COPYRIGHT"
KEYWORD_MAKE = "MAKE"
KEYWORD_MODEL = "MODEL"

# S3 INI
PROPERTY_SECTION_FILES = "FILES"
PROPERTY_SECTION_KEYS = "KEYS"
PROPERTY_IMAGES = "IMAGES"
PROPERTY_WEEDS_LOG = "WEEDS-LOG"
PROPERTY_KEY_IMAGES = "images.tar"
PROPERTY_KEY_WEEDS_LOG = "weeds-log"

#
# X M P P  R O O M S  A N D  J I D S
#
OLD_MESSAGE                          = 5000

PROPERTY_SECTION_XMPP                = "XMPP"
PROPERTY_SECTION_GENERAL             = "GENERAL"
PROPERTY_ROOT                        = "ROOT"
PROPERTY_PREFIX                      = "PREFIX"
PROPERTY_SUFFIX_META                 = "SUFFIX_META"
PROPERTY_DOMAIN_DNS                  = "DOMAIN"
PROPERTY_CONFERENCE_DNS              = "CONFERENCE_DNS"
PROPERTY_ROOM_ODOMETRY               = "ROOM_ODOMETRY"
PROPERTY_ROOM_TREATMENT              = "ROOM_TREATMENT"
PROPERTY_ROOM_SYSTEM                 = "ROOM_SYSTEM"
PROPERTY_SERVER                      = "SERVER"

PROPERTY_JID_ODOMETRY                = "JID_ODOMETRY"
PROPERTY_JID_RIO                     = "JID_RIO"
PROPERTY_JID_JETSON                  = "JID_JETSON"
PROPERTY_JID_JETSON_1                = "JID_JETSON1"
PROPERTY_JID_JETSON_2                = "JID_JETSON2"
PROPERTY_JID_CONSOLE                 = "JID_CONSOLE"
PROPERTY_JID_CLOUD                   = "JID_CLOUD"
PROPERTY_JID_CLOUD_1                 = "JID_CLOUD1"
PROPERTY_JID_CLOUD_2                 = "JID_CLOUD2"
PROPERTY_JID_CLOUD_3                 = "JID_CLOUD3"
PROPERTY_JID_CONTROL                 = "JID_CONTROL"
PROPERTY_POSITION                    = "POSITION"


PROPERTY_DEFAULT_PASSWORD            = "DEFAULT_PASSWORD"

PROPERTY_NICK_ODOMETRY               = "NICK_ODOMETRY"
PROPERTY_NICK_TREATMENT              = "NICK_TREATMENT"
PROPERTY_NICK_JETSON                 = "NICK_JETSON"
PROPERTY_NICK_JETSON_1               = "NICK_JETSON_1"
PROPERTY_NICK_JETSON_2               = "NICK_JETSON_2"
PROPERTY_NICK_CONSOLE                = "NICK_CONSOLE"
PROPERTY_NICK_CLOUD                  = "NICK_CLOUD"
PROPERTY_NICK_CLOUD_RIGHT            = "NICK_CLOUD_RIGHT"
PROPERTY_NICK_CLOUD_MIDDLE           = "NICK_CLOUD_MIDDLE"
PROPERTY_NICK_CLOUD_LEFT             = "NICK_CLOUD_LEFT"
PROPERTY_NICK_CONTROL                = "NICK_CONTROL"
PROPERTY_NICK_IMU                    = "NICK_IMU"
PROPERTY_MSG_TYPE_NORMAL             = "MSG_TYPE_NORMAL"
PROPERTY_MSG_TYPE_GROUPCHAT          = "MSG_TYPE_GROUPCHAT"

XMPP_PORT                            = 5222

PROPERTY_LEGO_A                      = "LEGO_A"
PROPERTY_LEGO_B                      = "LEGO_B"

PROPERTY_EMITTER_DIAG                = "DIAG-TYPE"
#
# J S O N  A N D  M E S S A G E S
#
JSON_ACTION         = "action"
JSON_DATA           = "data"
JSON_DIAG_MSG       = "diagnostic"
JSON_DISTANCE       = "distance"
JSON_IMAGE_NO       = "image_no"
JSON_SPEED          = "speed"
JSON_TIME           = "timestamp"
JSON_TOTAL_DISTANCE = "total_distance"
JSON_NAME           = "name"
JSON_PLAN           = "plan"
JSON_DIAG_RSLT      = "result"
JSON_DIAG_DETAIL    = "detail"
JSON_SOURCE         = "source"
JSON_LATITUDE       = "latitude"
JSON_LONGITUDE      = "longitude"
JSON_URL            = "url"
JSON_POSITION       = "position"
JSON_PULSES         = "pulses"
JSON_OPERATION      = "operation"
JSON_STATUS_CAMERA  = "status-camera"
JSON_GYRO           = "gyro"
JSON_ACCELERATION   = "acceleration"
JSON_PARAM_GSD      = "param-gsd"
JSON_PULSE_START    = "pulse_start"
JSON_PULSE_STOP     = "pulse_stop"
JSON_EMITTER_TIER   = "tier"
JSON_EMITTER_POS    = "position"
JSON_EMITTER_NUMBER = "number"
JSON_EMITTER_DURATION = "duration"
JSON_TARGET_LOCATION    = "location"
JSON_TARGET_DISTANCE    = "distance"

ENTITY_CAMERA       = "camera"
ENTITY_EMITTERS     = "emitters"
ENTITY_DAQ          = "daq"
ENTITY_SYSTEM       = "system"
ENTITY_GPS          = "gps"

EMITTER_ALL         = "0"
EMITTER_NOT_SET     = "-1"

SOURCE_VIRTUAL      = "virtual"
SOURCE_PHYSICAL     = "physical"

MSG_RAW             = "raw"

ACTION_START        = "START"
ACTION_STOP         = "STOP"
ACTION_CONFIGURE    = "CONFIGURE"

DIAG_PASS   = "pass"
DIAG_FAIL   = "fail"

EMITTER_DIAGNOSTIC_WET = "wet"
EMITTER_DIAGNOSTIC_DRY = "dry"

class Diagnostic(Enum):
    FAIL = 0
    OK = 1

class Forward(Enum):
    CLOCKWISE = 0
    COUNTER_CLOCKWISE = 1

class Treatment(Enum):
    RAW_IMAGE = 0
    PLAN = 1
    EMITTER_INSTRUCTIONS = 2
    CLASSIFICATION = 3
    UNKNOWN = 4

class Position(Enum):
    LEFT = 0
    MIDDLE = 1
    RIGHT = 2

class SubsystemType(Enum):
    VIRTUAL = 0
    PHYSICAL = 1

class ImageType(Enum):
    RGB = 0
    DEPTH = 1

class Action(Enum):
    START = 0
    STOP = 1
    START_DIAG = 2
    DIAG_REPORT = 3
    PING = 4
    ACK = 5
    CURRENT = 6
    CONFIGURE = 7

class Operation(Enum):
    QUIESCENT = 0
    IMAGING = 1
    WEEDING = 2

class Orientation(Enum):
    UP = 0
    DOWN = 1
    UNKNOWN = 2

class OperationalStatus(Enum):
    OK = 0
    UNKNOWN = 1
    FAIL = 2

class XMPPStatus(Enum):
    CONNECTED = 0
    DISCONNECT_RETRY = 1
    DISCONNECT_FAIL = 2


class Status(Enum):
    QUIESCENT = 0
    RUNNING = 1
    EXIT_OK = 2
    EXIT_FATAL = 3

class Capture(Enum):
    DEPTH = 0
    IMU = 1
    RGB = 2

class Side(Enum):
    RIGHT = 0
    LEFT = 1


