<?php
ini_set('display_errors', 0);
/**
 * Human ABX Experiment - Response Recording Backend
 *
 * Receives POST requests with JSON response data and saves to CSV files.
 *
 * Setup:
 * 1. Copy or symlink this file to ~/public_html/human_abx/save_responses.php
 * 2. Create a data directory: mkdir -p ~/public_html/human_abx/data
 * 3. Make it writable: chmod 755 ~/public_html/human_abx/data
 * 4. Set submit URL param: ?submit=https://yourserver.com/human_abx/save_responses.php
 * 5. Configure $completion_url below
 */

// ============== CONFIGURATION ==============
// Prolific completion URL - participants redirect here after successful submission
$completion_url = 'https://app.prolific.com/submissions/complete?cc=C7N8WBPR'; // e.g., 'https://app.prolific.com/submissions/complete?cc=XXXXXX'
// Prolific screen-out URL - participants who fail catch trials redirect here
$attention_fail_url = 'https://app.prolific.com/submissions/complete?cc=C13YLCJJ'; // e.g., 'https://app.prolific.com/submissions/complete?cc=SCREENOUT'
// ===========================================

header('Content-Type: application/json');
header('Access-Control-Allow-Origin: *');
header('Access-Control-Allow-Methods: POST, OPTIONS');
header('Access-Control-Allow-Headers: Content-Type');

// Handle preflight
if ($_SERVER['REQUEST_METHOD'] === 'OPTIONS') {
    http_response_code(200);
    exit;
}

if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
    http_response_code(405);
    echo json_encode(['error' => 'Method not allowed']);
    exit;
}

// Read JSON input
$input = file_get_contents('php://input');
$data = json_decode($input, true);

if (!$data) {
    http_response_code(400);
    echo json_encode(['error' => 'Invalid JSON']);
    exit;
}

// Extract fields
$participant_id = $data['participant_id'] ?? 'unknown';
$list_id = $data['list_id'] ?? 'unknown';
$prolific_pid = $data['prolific_pid'] ?? '';
$study_id = $data['study_id'] ?? '';
$session_id = $data['session_id'] ?? '';
$screened_out = $data['screened_out'] ?? false;
$responses = $data['responses'] ?? [];

if (empty($responses)) {
    http_response_code(400);
    echo json_encode(['error' => 'No responses']);
    exit;
}

// Sanitize for filename
$safe_participant = preg_replace('/[^a-zA-Z0-9_-]/', '_', $participant_id);
$safe_list = preg_replace('/[^a-zA-Z0-9_-]/', '_', $list_id);
$timestamp = date('Ymd_His');

// Data directory (relative to this script)
$data_dir = __DIR__ . '/data';
if (!is_dir($data_dir)) {
    mkdir($data_dir, 0755, true);
}

// Output filename
$filename = "{$data_dir}/responses_{$safe_participant}_{$safe_list}_{$timestamp}.csv";

// Get headers from first response
$headers = array_keys($responses[0]);
// Add prolific fields if not present
$extra_headers = ['prolific_pid', 'study_id', 'session_id'];
foreach ($extra_headers as $h) {
    if (!in_array($h, $headers)) {
        $headers[] = $h;
    }
}

// Write CSV
$fp = fopen($filename, 'w');
if (!$fp) {
    http_response_code(500);
    echo json_encode(['error' => 'Failed to create file']);
    exit;
}

fputcsv($fp, $headers);

foreach ($responses as $row) {
    // Add prolific fields
    $row['prolific_pid'] = $prolific_pid;
    $row['study_id'] = $study_id;
    $row['session_id'] = $session_id;

    $line = [];
    foreach ($headers as $h) {
        $val = $row[$h] ?? '';
        // Convert booleans/nulls to strings
        if (is_bool($val)) {
            $val = $val ? 'true' : 'false';
        } elseif (is_null($val)) {
            $val = '';
        }
        $line[] = $val;
    }
    fputcsv($fp, $line);
}

fclose($fp);

// Update assignment status
// Extract dataset and list name from list_id (e.g., "lists/stress/participant_000.csv")
$dataset = null;
$list_name = null;
if (preg_match('/lists\/([^\/]+)\/(participant_\d+)\.csv/', $list_id, $matches)) {
    $dataset = $matches[1];
    $list_name = $matches[2];  // e.g., "participant_000"
} elseif (preg_match('/^(participant_\d+)\.csv$/', $list_id, $matches)) {
    // Try to infer dataset from assignment files
    $list_name = $matches[1];
    foreach (['stress', 'pitch_accent', 'mandarin_tone'] as $ds) {
        $af = "{$data_dir}/assignments_{$ds}.json";
        if (file_exists($af)) {
            $a = json_decode(file_get_contents($af), true) ?: [];
            $pkey = $prolific_pid ?: $session_id ?: $participant_id;
            if (isset($a['participant_to_list'][$pkey])) {
                $dataset = $ds;
                $list_name = $a['participant_to_list'][$pkey];
                break;
            }
        }
    }
}

if ($dataset !== null && $list_name !== null) {
    $assignment_file = "{$data_dir}/assignments_{$dataset}.json";
    $lock_file = "{$data_dir}/assignments_{$dataset}.lock";

    $lock_fp = fopen($lock_file, 'w');
    flock($lock_fp, LOCK_EX);

    try {
        if (file_exists($assignment_file)) {
            $assignments = json_decode(file_get_contents($assignment_file), true) ?: [];

            $new_status = $screened_out ? 'screened_out' : 'completed';
            if (isset($assignments['lists'][$list_name])) {
                $assignments['lists'][$list_name]['status'] = $new_status;
                $assignments['lists'][$list_name]['finished_at'] = time();
            }

            file_put_contents($assignment_file, json_encode($assignments, JSON_PRETTY_PRINT), LOCK_EX);
        }
    } finally {
        flock($lock_fp, LOCK_UN);
        fclose($lock_fp);
    }
}

// Also append to master log
$master_log = "{$data_dir}/submissions.log";
$status = $screened_out ? 'SCREENED_OUT' : 'COMPLETED';
$log_entry = date('c') . "\t{$participant_id}\t{$list_id}\t{$prolific_pid}\t{$study_id}\t{$status}\t" . count($responses) . " responses\t{$filename}\n";
file_put_contents($master_log, $log_entry, FILE_APPEND | LOCK_EX);

$response = [
    'ok' => true,
    'participant_id' => $participant_id,
    'responses_count' => count($responses),
];

if (!empty($completion_url)) {
    $response['completion_url'] = $completion_url;
}
if (!empty($attention_fail_url)) {
    $response['attention_fail_url'] = $attention_fail_url;
}

echo json_encode($response);
