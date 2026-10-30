import { NextRequest, NextResponse } from "next/server";
import { getSessionFromRequest } from "@/lib/session-manager";

export async function GET(req: NextRequest) {
  const user = await getSessionFromRequest(req);

  if (!user) {
    return NextResponse.json(
      {
        authenticated: false,
        user: null,
      },
      { status: 401 }
    );
  }

  // Safe user information only
  return NextResponse.json({
    authenticated: true,
    user: {
      id: user.id,
      email: user.email,
      displayName: user.displayName,
      role: user.role,
      isEmailVerified: user.isEmailVerified,
      status: user.status,
    },
  });
}

