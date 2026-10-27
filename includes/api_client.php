<?php
declare(strict_types=1);

const BACKEND_URL = 'http://127.0.0.1:8000';

function api_request(string $path, ?array $payload = null): array {
    $url = BACKEND_URL . $path;
    $options = [
        'http' => [
            'method' => $payload === null ? 'GET' : 'POST',
            'timeout' => $payload === null ? 8 : 180,
            'ignore_errors' => true,
            'header' => "Accept: application/json\r\n" . ($payload === null ? '' : "Content-Type: application/json\r\n"),
            'content' => $payload === null ? '' : json_encode($payload, JSON_THROW_ON_ERROR),
        ],
    ];
    $context = stream_context_create($options);
    $body = @file_get_contents($url, false, $context);
    if ($body === false) {
        return ['ok' => false, 'error' => 'The prediction service is not responding. Start the Python backend and reload this page.'];
    }
    $decoded = json_decode($body, true);
    if (!is_array($decoded)) {
        return ['ok' => false, 'error' => 'The prediction service returned an unreadable response.'];
    }
    $status = 0;
    foreach ($http_response_header ?? [] as $header) {
        if (preg_match('/^HTTP\/\S+\s+(\d{3})/', $header, $match)) $status = (int)$match[1];
    }
    if ($status < 200 || $status >= 300) {
        $detail = $decoded['detail'] ?? 'The request could not be completed.';
        if (is_array($detail)) $detail = 'Please check the submitted values.';
        return ['ok' => false, 'error' => (string)$detail];
    }
    return ['ok' => true, 'data' => $decoded];
}
