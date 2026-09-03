<?php
ini_set('display_errors', 0);
/**
 * Human ABX Experiment - Smart Participant Assignment
 *
 * Assigns participants to participant lists, tracking status:
 * - available: No one assigned or previous participant screened out
 * - in_progress: Participant currently working
 * - completed: Participant finished successfully
 *
 * Usage:
 *   assign.php?dataset=stress&PROLIFIC_PID=xxx&STUDY_ID=yyy&SESSION_ID=zzz
 *
 * For testing (bypasses assignment):
 *   index.html?list=lists/stress/participant_000.csv
 */

// ============== CONFIGURATION ==============
$base_url = '';  // Auto-detect if empty
$submit_url = ''; // Auto-detect if empty

$valid_datasets = ['stress', 'pitch_accent', 'tone'];
$dataset_lists = [];  // dataset => [list_name, list_name, ...]
foreach ($valid_datasets as $ds) {
    $dir = __DIR__ . "/lists/{$ds}";
    $dataset_lists[$ds] = [];
    if (is_dir($dir)) {
        $files = glob("{$dir}/participant_*.csv");
        if ($files) {
            sort($files);  // Ensure consistent ordering
            foreach ($files as $f) {
                // Extract list name without extension (e.g., "participant_000")
                $dataset_lists[$ds][] = basename($f, '.csv');
            }
        }
    }
}

// Timeout: if a participant hasn't completed after this many seconds,
// consider their list available again
$in_progress_timeout_seconds = 3600; // 1 hour
// ===========================================

// Auto-detect URLs
if (empty($base_url)) {
    $protocol = (!empty($_SERVER['HTTPS']) && $_SERVER['HTTPS'] !== 'off') ? 'https' : 'http';
    $host = $_SERVER['HTTP_HOST'];
    $dir = dirname($_SERVER['SCRIPT_NAME']);
    $base_url = "{$protocol}://{$host}{$dir}";
}
if (empty($submit_url)) {
    $submit_url = "{$base_url}/save_responses.php";
}

// Get parameters
$dataset = $_GET['dataset'] ?? '';
$prolific_pid = $_GET['PROLIFIC_PID'] ?? '';
$participant_id = $_GET['participant_id'] ?? '';
$study_id = $_GET['STUDY_ID'] ?? '';
$session_id = $_GET['SESSION_ID'] ?? '';

// Validate dataset
if (!isset($dataset_lists[$dataset])) {
    http_response_code(400);
    die("Invalid or missing dataset. Valid options: " . implode(', ', array_keys($dataset_lists)));
}

$available_lists = $dataset_lists[$dataset];
if (empty($available_lists)) {
    http_response_code(503);
    die("No participant lists found for dataset: {$dataset}");
}

// Data directory
$data_dir = __DIR__ . '/data';
if (!is_dir($data_dir)) {
    mkdir($data_dir, 0755, true);
}

// Assignment tracking file
$assignment_file = "{$data_dir}/assignments_{$dataset}.json";

// Lock file for concurrent access
$lock_file = "{$data_dir}/assignments_{$dataset}.lock";
$lock_fp = fopen($lock_file, 'w');
flock($lock_fp, LOCK_EX);

try {
    // Load existing assignments
    $assignments = [];
    if (file_exists($assignment_file)) {
        $assignments = json_decode(file_get_contents($assignment_file), true) ?: [];
    }

    // Initialize lists structure if needed
    if (!isset($assignments['lists'])) {
        $assignments['lists'] = [];
    }

    // Ensure all current lists are in the assignments (handles new lists added later)
    foreach ($available_lists as $list_name) {
        if (!isset($assignments['lists'][$list_name])) {
            $assignments['lists'][$list_name] = ['status' => 'available'];
        }
    }

    // Participant key
    $participant_key = $prolific_pid ?: $participant_id ?: $session_id ?: uniqid('anon_');

    // Check if this participant already has an assignment
    $assigned_list = null;
    if (isset($assignments['participant_to_list'][$participant_key])) {
        $assigned_list = $assignments['participant_to_list'][$participant_key];
    } else {
        // Find an available list
        $now = time();

        // Priority 1: Lists that are available (never started or screened out)
        foreach ($available_lists as $list_name) {
            $list = $assignments['lists'][$list_name] ?? ['status' => 'available'];
            if ($list['status'] === 'available' || $list['status'] === 'screened_out') {
                $assigned_list = $list_name;
                break;
            }
        }

        // Priority 2: Lists that timed out (in_progress too long)
        if ($assigned_list === null) {
            foreach ($available_lists as $list_name) {
                $list = $assignments['lists'][$list_name] ?? ['status' => 'available'];
                if ($list['status'] === 'in_progress') {
                    $started = $list['started_at'] ?? 0;
                    if ($now - $started > $in_progress_timeout_seconds) {
                        $assigned_list = $list_name;
                        break;
                    }
                }
            }
        }

        // No available lists
        if ($assigned_list === null) {
            flock($lock_fp, LOCK_UN);
            fclose($lock_fp);
            http_response_code(503);
            die("All participant lists are currently in use. Please try again later.");
        }

        // Record assignment
        $assignments['lists'][$assigned_list] = [
            'status' => 'in_progress',
            'participant' => $participant_key,
            'started_at' => $now,
        ];
        $assignments['participant_to_list'][$participant_key] = $assigned_list;

        // Save
        file_put_contents($assignment_file, json_encode($assignments, JSON_PRETTY_PRINT), LOCK_EX);
    }

} finally {
    flock($lock_fp, LOCK_UN);
    fclose($lock_fp);
}

// Build redirect URL
$list_path = "lists/{$dataset}/{$assigned_list}.csv";

$params = [
    'list' => $list_path,
];

if ($prolific_pid) $params['PROLIFIC_PID'] = $prolific_pid;
if ($participant_id) $params['participant_id'] = $participant_id;
if ($study_id) $params['STUDY_ID'] = $study_id;
if ($session_id) $params['SESSION_ID'] = $session_id;

$redirect_url = "{$base_url}/index.html?" . http_build_query($params);

// Redirect
header("Location: {$redirect_url}");
exit;
