#!/bin/bash
export PATH="/Users/sheldonmrampton/.nvm/versions/node/v24.12.0/bin:$PATH"
exec /Users/sheldonmrampton/.nvm/versions/node/v24.12.0/bin/node /Users/sheldonmrampton/.nvm/versions/node/v24.12.0/lib/node_modules/@modelcontextprotocol/server-filesystem/dist/index.js "$@"
