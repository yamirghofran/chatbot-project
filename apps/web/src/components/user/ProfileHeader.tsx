import { Ellipsis } from "lucide-react";
import type { User } from "@/lib/types";
import { Avatar, AvatarImage, AvatarFallback } from "@/components/ui/avatar";
import { Button } from "@/components/ui/button";

export type ProfileHeaderProps = {
  user: User;
  isOwnProfile?: boolean;
  followingCount?: number;
  followersCount?: number;
  onFollow?: () => void;
  onEditProfile?: () => void;
};

function getInitials(name: string) {
  return name
    .split(" ")
    .map((w) => w[0])
    .slice(0, 2)
    .join("")
    .toUpperCase();
}

export function ProfileHeader({ user, isOwnProfile, followingCount = 0, followersCount = 0, onFollow, onEditProfile }: ProfileHeaderProps) {
  return (
    <div className="flex items-center gap-4">
      <Avatar size="xl">
        {user.avatarUrl && (
          <AvatarImage src={user.avatarUrl} alt={user.displayName} />
        )}
        <AvatarFallback>{getInitials(user.displayName)}</AvatarFallback>
      </Avatar>
      <div className="flex-1 min-w-0">
        <p className="font-heading text-lg font-semibold text-foreground truncate">
          {user.displayName}
        </p>
        <p className="text-sm text-muted-foreground truncate">@{user.handle}</p>
        <div className="flex gap-3 mt-1 text-sm">
          <span>
            <span className="font-semibold text-foreground">{followingCount}</span>{" "}
            <span className="text-muted-foreground">following</span>
          </span>
          <span>
            <span className="font-semibold text-foreground">{followersCount}</span>{" "}
            <span className="text-muted-foreground">followers</span>
          </span>
        </div>
      </div>
      <div className="flex items-center gap-2 shrink-0">
        {isOwnProfile ? (
          <Button variant="outline" size="sm" onClick={onEditProfile}>
            Edit Profile
          </Button>
        ) : (
          <>
            <Button variant="outline" size="sm" onClick={onFollow}>
              Follow
            </Button>
            <Button variant="outline" size="icon-sm" aria-label="More actions">
              <Ellipsis />
            </Button>
          </>
        )}
      </div>
    </div>
  );
}
